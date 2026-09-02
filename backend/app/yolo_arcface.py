import os
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import cv2
import numpy as np
import onnxruntime as ort
from insightface.utils import face_align


YOLO_MODEL_PATH = os.getenv("YOLO_FACE_MODEL_PATH", "/app/models/yolov8n-face.onnx")
ARCFACE_MODEL_PATH = os.getenv(
    "ARCFACE_MODEL_PATH", "/app/models/arcface-r50-w600k.onnx"
)
YOLO_INPUT_SIZE = int(os.getenv("YOLO_FACE_INPUT_SIZE", "640"))
YOLO_CONFIDENCE_THRESHOLD = float(os.getenv("YOLO_FACE_CONFIDENCE", "0.25"))
YOLO_NMS_THRESHOLD = float(os.getenv("YOLO_FACE_NMS_THRESHOLD", "0.45"))
YOLO_MIN_FACE_ASPECT_RATIO = float(
    os.getenv("YOLO_FACE_MIN_ASPECT_RATIO", "0.40")
)
YOLO_MAX_FACE_ASPECT_RATIO = float(
    os.getenv("YOLO_FACE_MAX_ASPECT_RATIO", "1.80")
)
ARCFACE_BATCH_SIZE = max(1, int(os.getenv("ARCFACE_BATCH_SIZE", "16")))
TENSORRT_CACHE_ROOT = os.getenv("TENSORRT_CACHE_ROOT", "/app/tensorrt-cache")

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnalyzedFace:
    bbox: np.ndarray
    kps: np.ndarray
    det_score: float
    normed_embedding: np.ndarray


@dataclass(frozen=True)
class DetectedYoloFace:
    bbox: np.ndarray
    landmarks: np.ndarray
    confidence: float


Provider = Union[str, Tuple[str, Dict[str, str]]]


def _environment_enabled(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _execution_providers(
    model_name: str,
) -> List[Provider]:
    available = ort.get_available_providers()
    providers: List[Provider] = []
    tensorrt_enabled = _environment_enabled("FACE_ENABLE_TENSORRT", "true")
    if model_name == "arcface_r50":
        tensorrt_enabled = _environment_enabled(
            "ARCFACE_ENABLE_TENSORRT", "false"
        )
    if (
        tensorrt_enabled
        and "TensorrtExecutionProvider" in available
    ):
        cache_path = str(Path(TENSORRT_CACHE_ROOT) / model_name)
        Path(cache_path).mkdir(parents=True, exist_ok=True)
        options = {
            "trt_engine_cache_enable": "True",
            "trt_engine_cache_path": cache_path,
            "trt_fp16_enable": (
                "True"
                if _environment_enabled("TENSORRT_FP16_ENABLE", "false")
                else "False"
            ),
        }
        providers.append(("TensorrtExecutionProvider", options))
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers


def _create_session(
    model_path: str,
    model_name: str,
    session_options: ort.SessionOptions,
) -> ort.InferenceSession:
    providers = _execution_providers(model_name)
    try:
        session = ort.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=providers,
        )
    except Exception:
        logger.exception(
            "TensorRT session could not be created for %s; CUDA fallback is used.",
            model_name,
        )
        fallback = [
            provider
            for provider in ("CUDAExecutionProvider", "CPUExecutionProvider")
            if provider == "CPUExecutionProvider"
            or provider in ort.get_available_providers()
        ]
        session = ort.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=fallback,
        )
    logger.info("%s execution providers: %s", model_name, session.get_providers())
    return session


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _softmax(values: np.ndarray, axis: int) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=axis, keepdims=True)


def _has_plausible_face_geometry(box: np.ndarray) -> bool:
    width = max(0.0, float(box[2] - box[0]))
    height = max(0.0, float(box[3] - box[1]))
    if width <= 0.0 or height <= 0.0:
        return False
    aspect_ratio = width / height
    return YOLO_MIN_FACE_ASPECT_RATIO <= aspect_ratio <= YOLO_MAX_FACE_ASPECT_RATIO


class YoloV8FaceDetector:
    def __init__(self) -> None:
        session_options = ort.SessionOptions()
        session_options.log_severity_level = 3
        self.session = _create_session(
            YOLO_MODEL_PATH,
            "yolov8_face",
            session_options,
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

    def detect(self, image: np.ndarray) -> List[DetectedYoloFace]:
        tensor, scale, pad_x, pad_y = self._prepare_image(image)
        outputs = self.session.run(self.output_names, {self.input_name: tensor})
        boxes, scores, landmarks = self._decode(outputs)
        if not boxes:
            return []

        nms_boxes = [
            [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]
            for x1, y1, x2, y2 in boxes
        ]
        selected = cv2.dnn.NMSBoxes(
            nms_boxes,
            scores,
            YOLO_CONFIDENCE_THRESHOLD,
            YOLO_NMS_THRESHOLD,
        )
        if len(selected) == 0:
            return []

        image_height, image_width = image.shape[:2]
        results: List[DetectedYoloFace] = []
        for raw_index in np.asarray(selected).reshape(-1):
            index = int(raw_index)
            box = np.asarray(boxes[index], dtype=np.float32)
            box[[0, 2]] = (box[[0, 2]] - pad_x) / scale
            box[[1, 3]] = (box[[1, 3]] - pad_y) / scale
            box[[0, 2]] = np.clip(box[[0, 2]], 0, image_width)
            box[[1, 3]] = np.clip(box[[1, 3]], 0, image_height)

            points = np.asarray(landmarks[index], dtype=np.float32).copy()
            points[:, 0] = (points[:, 0] - pad_x) / scale
            points[:, 1] = (points[:, 1] - pad_y) / scale
            points[:, 0] = np.clip(points[:, 0], 0, image_width)
            points[:, 1] = np.clip(points[:, 1], 0, image_height)

            if not _has_plausible_face_geometry(box):
                continue
            results.append(
                DetectedYoloFace(
                    bbox=box,
                    landmarks=points,
                    confidence=float(scores[index]),
                )
            )

        return sorted(results, key=lambda face: (float(face.bbox[0]), float(face.bbox[1])))

    def providers(self) -> List[str]:
        return self.session.get_providers()

    def _prepare_image(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, float, int, int]:
        image_height, image_width = image.shape[:2]
        scale = min(YOLO_INPUT_SIZE / image_width, YOLO_INPUT_SIZE / image_height)
        resized_width = max(1, int(round(image_width * scale)))
        resized_height = max(1, int(round(image_height * scale)))
        resized = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )
        pad_x = (YOLO_INPUT_SIZE - resized_width) // 2
        pad_y = (YOLO_INPUT_SIZE - resized_height) // 2
        canvas = np.full(
            (YOLO_INPUT_SIZE, YOLO_INPUT_SIZE, 3),
            114,
            dtype=np.uint8,
        )
        canvas[
            pad_y : pad_y + resized_height,
            pad_x : pad_x + resized_width,
        ] = resized
        tensor = canvas[:, :, ::-1].transpose(2, 0, 1)
        tensor = np.ascontiguousarray(tensor, dtype=np.float32) / 255.0
        return tensor[None, ...], scale, pad_x, pad_y

    def _decode(
        self, outputs: Sequence[np.ndarray]
    ) -> Tuple[List[List[float]], List[float], List[np.ndarray]]:
        boxes: List[List[float]] = []
        scores: List[float] = []
        landmarks: List[np.ndarray] = []
        projection = np.arange(16, dtype=np.float32)
        ordered_outputs = sorted(outputs, key=lambda output: output.shape[2], reverse=True)
        for stride, output in zip((8, 16, 32), ordered_outputs):
            _, channel_count, grid_height, grid_width = output.shape
            if channel_count != 80:
                raise RuntimeError(
                    f"Unexpected YOLOv8-Face output channel count: {channel_count}"
                )

            rows = output[0].transpose(1, 2, 0).reshape(-1, channel_count)
            confidence = _sigmoid(rows[:, 64])
            candidate_indices = np.flatnonzero(
                confidence >= YOLO_CONFIDENCE_THRESHOLD
            )
            if candidate_indices.size == 0:
                continue

            grid_y, grid_x = np.meshgrid(
                np.arange(grid_height, dtype=np.float32),
                np.arange(grid_width, dtype=np.float32),
                indexing="ij",
            )
            anchor_x = grid_x.reshape(-1) + 0.5
            anchor_y = grid_y.reshape(-1) + 0.5

            candidate_rows = rows[candidate_indices]
            distances = _softmax(
                candidate_rows[:, :64].reshape(-1, 4, 16),
                axis=2,
            ) @ projection
            selected_anchor_x = anchor_x[candidate_indices]
            selected_anchor_y = anchor_y[candidate_indices]
            x1 = (selected_anchor_x - distances[:, 0]) * stride
            y1 = (selected_anchor_y - distances[:, 1]) * stride
            x2 = (selected_anchor_x + distances[:, 2]) * stride
            y2 = (selected_anchor_y + distances[:, 3]) * stride

            raw_landmarks = candidate_rows[:, 65:].reshape(-1, 5, 3)
            decoded_landmarks = np.empty((len(candidate_indices), 5, 2), dtype=np.float32)
            decoded_landmarks[:, :, 0] = (
                raw_landmarks[:, :, 0] * 2.0
                + selected_anchor_x[:, None]
                - 0.5
            ) * stride
            decoded_landmarks[:, :, 1] = (
                raw_landmarks[:, :, 1] * 2.0
                + selected_anchor_y[:, None]
                - 0.5
            ) * stride

            boxes.extend(np.stack((x1, y1, x2, y2), axis=1).tolist())
            scores.extend(confidence[candidate_indices].astype(float).tolist())
            landmarks.extend(decoded_landmarks)

        return boxes, scores, landmarks


class ArcFaceR50Recognizer:
    def __init__(self) -> None:
        session_options = ort.SessionOptions()
        session_options.log_severity_level = 3
        self.session = _create_session(
            ARCFACE_MODEL_PATH,
            "arcface_r50",
            session_options,
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def embed(
        self,
        image: np.ndarray,
        faces: Sequence[DetectedYoloFace],
    ) -> np.ndarray:
        if not faces:
            return np.empty((0, 512), dtype=np.float32)

        aligned_faces = [self.align(image, face) for face in faces]
        return self.embed_aligned(aligned_faces)

    def align(self, image: np.ndarray, face: DetectedYoloFace) -> np.ndarray:
        return face_align.norm_crop(
            image,
            landmark=face.landmarks,
            image_size=112,
        )

    def embed_aligned(self, aligned_faces: Sequence[np.ndarray]) -> np.ndarray:
        if not aligned_faces:
            return np.empty((0, 512), dtype=np.float32)

        batches: List[np.ndarray] = []
        for offset in range(0, len(aligned_faces), ARCFACE_BATCH_SIZE):
            batch_faces = aligned_faces[offset : offset + ARCFACE_BATCH_SIZE]
            tensor = cv2.dnn.blobFromImages(
                batch_faces,
                scalefactor=1.0 / 127.5,
                size=(112, 112),
                mean=(127.5, 127.5, 127.5),
                swapRB=True,
            )
            embeddings = self.session.run(
                [self.output_name],
                {self.input_name: tensor},
            )[0].astype(np.float32)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            batches.append(embeddings / np.maximum(norms, 1e-12))
        return np.concatenate(batches, axis=0)

    def providers(self) -> List[str]:
        return self.session.get_providers()


class YoloArcFaceEngine:
    def __init__(self) -> None:
        self.detector = YoloV8FaceDetector()
        self.recognizer = ArcFaceR50Recognizer()

    def detect(self, image: np.ndarray) -> List[DetectedYoloFace]:
        return self.detector.detect(image)

    def get(self, image: np.ndarray) -> List[AnalyzedFace]:
        faces = self.detector.detect(image)
        embeddings = self.recognizer.embed(image, faces)
        return [
            AnalyzedFace(
                bbox=face.bbox,
                kps=face.landmarks,
                det_score=face.confidence,
                normed_embedding=embedding,
            )
            for face, embedding in zip(faces, embeddings)
        ]

    def providers(self) -> List[str]:
        detector_providers = self.detector.providers()
        recognizer_providers = self.recognizer.providers()
        return [
            provider
            for provider in detector_providers
            if provider in recognizer_providers
        ]


@lru_cache(maxsize=1)
def get_yolo_arcface_engine() -> YoloArcFaceEngine:
    return YoloArcFaceEngine()
