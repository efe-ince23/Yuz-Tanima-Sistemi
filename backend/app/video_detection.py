from dataclasses import dataclass, field
import time
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from app.video_frames import (
    SampledVideoFrame,
    VideoSamplingSummary,
    sample_video_object,
)
from app.video_config import get_video_settings
from app.yolo_arcface import get_yolo_arcface_engine


PixelBoundingBox = Tuple[int, int, int, int]
NormalizedBoundingBox = Tuple[float, float, float, float]
FaceLandmarks = Tuple[Tuple[float, float], ...]


@dataclass(frozen=True)
class VideoFaceDetection:
    face_index: int
    confidence: float
    bounding_box: PixelBoundingBox
    normalized_bounding_box: NormalizedBoundingBox
    landmarks: FaceLandmarks
    embedding: Optional[np.ndarray] = field(
        default=None,
        repr=False,
        compare=False,
    )


@dataclass(frozen=True)
class DetectedVideoFrame:
    frame_number: int
    timestamp_ms: int
    width: int
    height: int
    image: np.ndarray
    faces: Tuple[VideoFaceDetection, ...]


@dataclass(frozen=True)
class VideoDetectionSummary:
    source_fps: float
    source_frame_count: Optional[int]
    decoded_frame_count: int
    sampled_frame_count: int
    frames_with_faces: int
    detected_face_count: int
    width: int
    height: int
    last_timestamp_ms: int
    sampling_seconds: float = 0.0
    inference_seconds: float = 0.0
    detector_seconds: float = 0.0
    recognizer_seconds: float = 0.0


DetectionFrameHandler = Callable[[DetectedVideoFrame], None]


def _clamp_box(
    raw_box: np.ndarray,
    width: int,
    height: int,
) -> PixelBoundingBox:
    raw_x1, raw_y1, raw_x2, raw_y2 = (int(value) for value in raw_box)
    x1 = min(max(raw_x1, 0), width)
    y1 = min(max(raw_y1, 0), height)
    x2 = min(max(raw_x2, x1), width)
    y2 = min(max(raw_y2, y1), height)
    return x1, y1, x2, y2


def _normalize_box(
    box: PixelBoundingBox,
    width: int,
    height: int,
) -> NormalizedBoundingBox:
    x1, y1, x2, y2 = box
    return (
        round(x1 / width, 6),
        round(y1 / height, 6),
        round(x2 / width, 6),
        round(y2 / height, 6),
    )


def _build_detected_frame(
    sampled_frame: SampledVideoFrame,
    raw_faces,
    embeddings: Sequence[np.ndarray],
) -> DetectedVideoFrame:
    image = sampled_frame.image
    height, width = image.shape[:2]
    if len(embeddings) != len(raw_faces):
        raise RuntimeError("ArcFace tum tespit edilen yuzler icin vektor uretemedi.")
    faces = []
    for face_index, (raw_face, embedding) in enumerate(zip(raw_faces, embeddings)):
        bounding_box = _clamp_box(raw_face.bbox, width, height)
        if bounding_box[2] <= bounding_box[0] or bounding_box[3] <= bounding_box[1]:
            continue
        landmarks = tuple(
            (
                round(min(max(float(point[0]), 0.0), float(width)), 4),
                round(min(max(float(point[1]), 0.0), float(height)), 4),
            )
            for point in raw_face.landmarks
        )
        faces.append(
            VideoFaceDetection(
                face_index=face_index,
                confidence=round(float(raw_face.confidence), 6),
                bounding_box=bounding_box,
                normalized_bounding_box=_normalize_box(
                    bounding_box,
                    width,
                    height,
                ),
                landmarks=landmarks,
                embedding=np.asarray(embedding, dtype=np.float32).copy(),
            )
        )

    return DetectedVideoFrame(
        frame_number=sampled_frame.frame_number,
        timestamp_ms=sampled_frame.timestamp_ms,
        width=width,
        height=height,
        image=image,
        faces=tuple(faces),
    )


def _detect_frame_batch(
    sampled_frames: Sequence[SampledVideoFrame],
) -> Tuple[Tuple[DetectedVideoFrame, ...], float, float]:
    if not sampled_frames:
        return (), 0.0, 0.0
    engine = get_yolo_arcface_engine()
    faces_by_frame = []
    aligned_faces: List[np.ndarray] = []
    detector_started = time.perf_counter()
    for sampled_frame in sampled_frames:
        raw_faces = sorted(
            engine.detect(sampled_frame.image),
            key=lambda face: (float(face.bbox[0]), float(face.bbox[1])),
        )
        faces_by_frame.append(raw_faces)
        aligned_faces.extend(
            engine.recognizer.align(sampled_frame.image, face)
            for face in raw_faces
        )
    detector_seconds = time.perf_counter() - detector_started

    recognizer_started = time.perf_counter()
    if aligned_faces:
        embeddings = engine.recognizer.embed_aligned(aligned_faces)
    else:
        embeddings = np.empty((0, 512), dtype=np.float32)
    if len(embeddings) != len(aligned_faces):
        raise RuntimeError("ArcFace tum video batch yuzleri icin vektor uretemedi.")
    recognizer_seconds = time.perf_counter() - recognizer_started

    detected_frames = []
    offset = 0
    for sampled_frame, raw_faces in zip(sampled_frames, faces_by_frame):
        face_count = len(raw_faces)
        detected_frames.append(
            _build_detected_frame(
                sampled_frame,
                raw_faces,
                embeddings[offset : offset + face_count],
            )
        )
        offset += face_count
    return tuple(detected_frames), detector_seconds, recognizer_seconds


def _detect_frame(sampled_frame: SampledVideoFrame) -> DetectedVideoFrame:
    return _detect_frame_batch((sampled_frame,))[0][0]


def detect_video_faces(
    object_path: str,
    sample_fps: float,
    frame_handler: Optional[DetectionFrameHandler] = None,
) -> VideoDetectionSummary:
    frames_with_faces = 0
    detected_face_count = 0
    inference_seconds = 0.0
    detector_seconds = 0.0
    recognizer_seconds = 0.0
    pending_frames: List[SampledVideoFrame] = []
    batch_size = get_video_settings().arcface_frame_batch_size

    def flush_pending_frames() -> None:
        nonlocal frames_with_faces, detected_face_count
        nonlocal inference_seconds, detector_seconds, recognizer_seconds
        inference_started = time.perf_counter()
        detected_frames, batch_detector_seconds, batch_recognizer_seconds = (
            _detect_frame_batch(tuple(pending_frames))
        )
        inference_seconds += time.perf_counter() - inference_started
        detector_seconds += batch_detector_seconds
        recognizer_seconds += batch_recognizer_seconds
        for detected_frame in detected_frames:
            if detected_frame.faces:
                frames_with_faces += 1
                detected_face_count += len(detected_frame.faces)
            if frame_handler is not None:
                frame_handler(detected_frame)
        pending_frames.clear()

    def handle_sampled_frame(sampled_frame: SampledVideoFrame) -> None:
        pending_frames.append(sampled_frame)
        if len(pending_frames) >= batch_size:
            flush_pending_frames()

    sampling_started = time.perf_counter()
    sampling: VideoSamplingSummary = sample_video_object(
        object_path,
        sample_fps,
        handle_sampled_frame,
    )
    flush_pending_frames()
    sampling_seconds = time.perf_counter() - sampling_started
    return VideoDetectionSummary(
        source_fps=sampling.source_fps,
        source_frame_count=sampling.source_frame_count,
        decoded_frame_count=sampling.decoded_frame_count,
        sampled_frame_count=sampling.sampled_frame_count,
        frames_with_faces=frames_with_faces,
        detected_face_count=detected_face_count,
        width=sampling.width,
        height=sampling.height,
        last_timestamp_ms=sampling.last_timestamp_ms,
        sampling_seconds=sampling_seconds,
        inference_seconds=inference_seconds,
        detector_seconds=detector_seconds,
        recognizer_seconds=recognizer_seconds,
    )
