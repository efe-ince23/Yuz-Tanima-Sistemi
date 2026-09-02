import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis

from app.yolo_arcface import get_yolo_arcface_engine


MODEL_NAME = os.getenv("FACE_MODEL_NAME", "buffalo_sc")
FACE_ENGINE = os.getenv("FACE_ENGINE", "buffalo").strip().lower()
MATCH_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", "0.45"))
ANONYMOUS_MATCH_THRESHOLD = float(
    os.getenv("FACE_ANONYMOUS_MATCH_THRESHOLD", str(MATCH_THRESHOLD))
)
DUPLICATE_FACE_THRESHOLD = float(os.getenv("FACE_DUPLICATE_THRESHOLD", "0.55"))
PRIMARY_FACE_DOMINANCE_RATIO = float(
    os.getenv("PRIMARY_FACE_DOMINANCE_RATIO", "2.0")
)


@dataclass(frozen=True)
class PrimaryFaceData:
    embedding: np.ndarray
    confidence: float
    detected_face_count: int
    ignored_face_count: int


@dataclass(frozen=True)
class DetectedFaceData:
    embedding: Optional[np.ndarray]
    confidence: float
    bounding_box: Tuple[int, int, int, int]


class FaceCountError(ValueError):
    def __init__(self, face_count: int) -> None:
        self.face_count = face_count
        super().__init__("Exactly one face is required.")


class AmbiguousFacesError(FaceCountError):
    pass


@lru_cache(maxsize=1)
def get_face_analyzer() -> FaceAnalysis:
    available_providers = ort.get_available_providers()
    providers = ["CPUExecutionProvider"]
    context_id = -1

    if "CUDAExecutionProvider" in available_providers:
        providers.insert(0, "CUDAExecutionProvider")
        context_id = 0

    analyzer = FaceAnalysis(
        name=MODEL_NAME,
        allowed_modules=["detection", "recognition"],
        providers=providers,
    )
    analyzer.prepare(ctx_id=context_id, det_size=(640, 640))
    return analyzer


def _analyze_faces(image: np.ndarray) -> List[object]:
    if FACE_ENGINE == "yolo_arcface":
        return list(get_yolo_arcface_engine().get(image))
    if FACE_ENGINE != "buffalo":
        raise RuntimeError(f"Unsupported face engine: {FACE_ENGINE}")
    return list(get_face_analyzer().get(image))


def decode_image(content: bytes) -> Optional[np.ndarray]:
    encoded_image = np.frombuffer(content, dtype=np.uint8)
    return cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)


def detect_faces(image: np.ndarray) -> List[Dict[str, object]]:
    if FACE_ENGINE == "yolo_arcface":
        detected_faces = get_yolo_arcface_engine().detect(image)
    elif FACE_ENGINE == "buffalo":
        detected_faces = get_face_analyzer().get(image)
    else:
        raise RuntimeError(f"Unsupported face engine: {FACE_ENGINE}")
    image_height, image_width = image.shape[:2]
    results: List[Dict[str, object]] = []

    ordered_faces = sorted(
        detected_faces,
        key=lambda face: (float(face.bbox[0]), float(face.bbox[1])),
    )
    for face_index, face in enumerate(ordered_faces):
        raw_x1, raw_y1, raw_x2, raw_y2 = (int(value) for value in face.bbox)
        x1 = min(max(raw_x1, 0), image_width)
        y1 = min(max(raw_y1, 0), image_height)
        x2 = min(max(raw_x2, x1), image_width)
        y2 = min(max(raw_y2, y1), image_height)
        results.append(
            {
                "face_index": face_index,
                "bounding_box": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "width": x2 - x1,
                    "height": y2 - y1,
                },
                "confidence": round(
                    float(
                        face.confidence
                        if FACE_ENGINE == "yolo_arcface"
                        else face.det_score
                    ),
                    4,
                ),
            }
        )

    return results


def extract_all_faces_data(image: np.ndarray) -> List[DetectedFaceData]:
    detected_faces = _analyze_faces(image)
    image_height, image_width = image.shape[:2]
    ordered_faces = sorted(
        detected_faces,
        key=lambda face: (float(face.bbox[0]), float(face.bbox[1])),
    )
    results: List[DetectedFaceData] = []

    for face in ordered_faces:
        raw_x1, raw_y1, raw_x2, raw_y2 = (int(value) for value in face.bbox)
        x1 = min(max(raw_x1, 0), image_width)
        y1 = min(max(raw_y1, 0), image_height)
        x2 = min(max(raw_x2, x1), image_width)
        y2 = min(max(raw_y2, y1), image_height)
        embedding = face.normed_embedding
        results.append(
            DetectedFaceData(
                embedding=(
                    embedding.astype(np.float32) if embedding is not None else None
                ),
                confidence=float(face.det_score),
                bounding_box=(x1, y1, x2, y2),
            )
        )

    return results


def extract_single_face_data(image: np.ndarray) -> Tuple[np.ndarray, float]:
    faces = _analyze_faces(image)
    if len(faces) != 1:
        raise FaceCountError(len(faces))

    embedding = faces[0].normed_embedding
    if embedding is None:
        raise RuntimeError("Face embedding could not be generated.")

    return embedding.astype(np.float32), float(faces[0].det_score)


def extract_primary_face_data(image: np.ndarray) -> PrimaryFaceData:
    faces = _analyze_faces(image)
    face_count = len(faces)
    if face_count == 0:
        raise FaceCountError(0)

    selected_face = faces[0]
    if face_count > 1:
        faces_by_area = sorted(
            faces,
            key=lambda face: max(0.0, float(face.bbox[2] - face.bbox[0]))
            * max(0.0, float(face.bbox[3] - face.bbox[1])),
            reverse=True,
        )
        largest_face = faces_by_area[0]
        second_face = faces_by_area[1]
        largest_area = max(0.0, float(largest_face.bbox[2] - largest_face.bbox[0])) * max(
            0.0, float(largest_face.bbox[3] - largest_face.bbox[1])
        )
        second_area = max(0.0, float(second_face.bbox[2] - second_face.bbox[0])) * max(
            0.0, float(second_face.bbox[3] - second_face.bbox[1])
        )
        if second_area > 0 and largest_area / second_area < PRIMARY_FACE_DOMINANCE_RATIO:
            raise AmbiguousFacesError(face_count)
        selected_face = largest_face

    embedding = selected_face.normed_embedding
    if embedding is None:
        raise RuntimeError("Face embedding could not be generated.")

    return PrimaryFaceData(
        embedding=embedding.astype(np.float32),
        confidence=float(selected_face.det_score),
        detected_face_count=face_count,
        ignored_face_count=max(0, face_count - 1),
    )


def extract_single_embedding(image: np.ndarray) -> np.ndarray:
    embedding, _ = extract_single_face_data(image)
    return embedding


def compare_embeddings(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
    similarity = float(np.dot(embedding_a, embedding_b))
    return float(np.clip(similarity, -1.0, 1.0))


def available_execution_providers() -> List[str]:
    return ort.get_available_providers()


def active_execution_providers() -> List[str]:
    if FACE_ENGINE == "yolo_arcface":
        return get_yolo_arcface_engine().providers()
    analyzer = get_face_analyzer()
    detection_model = analyzer.models["detection"]
    return detection_model.session.get_providers()
