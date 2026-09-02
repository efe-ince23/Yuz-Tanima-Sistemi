import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple


def _positive_float(name: str, default: str) -> float:
    raw_value = os.getenv(name, default)
    try:
        value = float(raw_value)
    except ValueError as error:
        raise RuntimeError(f"{name} must be a number.") from error
    if value <= 0:
        raise RuntimeError(f"{name} must be greater than zero.")
    return value


def _positive_int(name: str, default: str) -> int:
    raw_value = os.getenv(name, default)
    try:
        value = int(raw_value)
    except ValueError as error:
        raise RuntimeError(f"{name} must be an integer.") from error
    if value <= 0:
        raise RuntimeError(f"{name} must be greater than zero.")
    return value


def _bounded_float(
    name: str,
    default: str,
    minimum: float,
    maximum: float,
) -> float:
    raw_value = os.getenv(name, default)
    try:
        value = float(raw_value)
    except ValueError as error:
        raise RuntimeError(f"{name} must be a number.") from error
    if not minimum <= value <= maximum:
        raise RuntimeError(
            f"{name} must be between {minimum} and {maximum}."
        )
    return value


def _normalized_values(name: str, default: str) -> Tuple[str, ...]:
    values = tuple(
        dict.fromkeys(
            item.strip().lower()
            for item in os.getenv(name, default).split(",")
            if item.strip()
        )
    )
    if not values:
        raise RuntimeError(f"{name} must contain at least one value.")
    return values


def _choice(name: str, default: str, allowed: Tuple[str, ...]) -> str:
    value = os.getenv(name, default).strip().lower()
    if value not in allowed:
        raise RuntimeError(f"{name} must be one of: {', '.join(allowed)}.")
    return value


def _boolean(name: str, default: str) -> bool:
    value = os.getenv(name, default).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be a boolean.")


@dataclass(frozen=True)
class VideoSettings:
    max_size_bytes: int
    max_duration_seconds: float
    sample_fps: float
    processing_concurrency: int
    allowed_content_types: Tuple[str, ...]
    allowed_containers: Tuple[str, ...]
    allowed_codecs: Tuple[str, ...]
    arcface_frame_batch_size: int = 1

    def supports_content_type(self, content_type: str) -> bool:
        normalized = content_type.lower().split(";", 1)[0].strip()
        return normalized in self.allowed_content_types

    def supports_container(self, container: str) -> bool:
        return container.strip().lower() in self.allowed_containers

    def supports_codec(self, codec: str) -> bool:
        return codec.strip().lower() in self.allowed_codecs


@dataclass(frozen=True)
class VideoTrackingSettings:
    max_gap_ms: int
    min_iou: float
    max_center_distance: float
    max_area_ratio: float
    engine: str = "custom"
    bytetrack_activation_threshold: float = 0.15
    bytetrack_matching_threshold: float = 0.90
    bytetrack_minimum_consecutive_frames: int = 1
    scene_cut_enabled: bool = True
    scene_cut_histogram_correlation: float = 0.35
    scene_cut_pixel_difference: float = 0.10
    appearance_enabled: bool = True
    appearance_min_similarity: float = 0.20
    appearance_weight: float = 0.45
    appearance_momentum: float = 0.80


@dataclass(frozen=True)
class VideoRecognitionSettings:
    samples_per_track: int
    anonymous_min_observations: int
    anonymous_min_quality_samples: int
    anonymous_min_quality: float
    anonymous_min_frontal_score: float
    anonymous_min_duration_ms: int
    anonymous_min_detection_confidence: float
    anonymous_min_face_size_px: int
    anonymous_min_sharpness: float
    anonymous_min_embedding_consistency: float
    anonymous_single_min_quality: float
    anonymous_single_min_frontal_score: float
    anonymous_single_min_detection_confidence: float
    anonymous_single_min_face_size_px: int
    anonymous_single_min_sharpness: float
    anonymous_short_min_observations: int
    anonymous_short_min_duration_ms: int
    window_ms: int
    strong_match_margin: float
    weak_known_min_duration_ms: int
    short_known_min_detection_confidence: float
    bridge_window_count: int
    continuity_window_count: int
    continuity_threshold: float
    continuity_distance_penalty: float
    reid_gallery_size: int
    reid_min_similarity: float


@lru_cache(maxsize=1)
def get_video_settings() -> VideoSettings:
    max_size_mb = _positive_int("VIDEO_MAX_SIZE_MB", "200")
    return VideoSettings(
        max_size_bytes=max_size_mb * 1024 * 1024,
        max_duration_seconds=_positive_float(
            "VIDEO_MAX_DURATION_SECONDS", "300"
        ),
        sample_fps=_positive_float("VIDEO_SAMPLE_FPS", "6"),
        processing_concurrency=_positive_int(
            "VIDEO_PROCESSING_CONCURRENCY", "1"
        ),
        allowed_content_types=_normalized_values(
            "VIDEO_ALLOWED_CONTENT_TYPES", "video/mp4"
        ),
        allowed_containers=_normalized_values(
            "VIDEO_ALLOWED_CONTAINERS", "mp4"
        ),
        allowed_codecs=_normalized_values("VIDEO_ALLOWED_CODECS", "h264"),
        arcface_frame_batch_size=_positive_int(
            "VIDEO_ARCFACE_FRAME_BATCH_SIZE", "1"
        ),
    )


@lru_cache(maxsize=1)
def get_video_tracking_settings() -> VideoTrackingSettings:
    max_gap_seconds = _positive_float("VIDEO_TRACK_MAX_GAP_SECONDS", "0.75")
    return VideoTrackingSettings(
        max_gap_ms=round(max_gap_seconds * 1000),
        min_iou=_bounded_float("VIDEO_TRACK_MIN_IOU", "0.10", 0.0, 1.0),
        max_center_distance=_bounded_float(
            "VIDEO_TRACK_MAX_CENTER_DISTANCE", "0.25", 0.01, 1.0
        ),
        max_area_ratio=_positive_float("VIDEO_TRACK_MAX_AREA_RATIO", "3.0"),
        engine=_choice(
            "VIDEO_TRACKER_ENGINE",
            "custom",
            ("custom", "bytetrack"),
        ),
        bytetrack_activation_threshold=_bounded_float(
            "VIDEO_BYTETRACK_ACTIVATION_THRESHOLD", "0.15", 0.0, 1.0
        ),
        bytetrack_matching_threshold=_bounded_float(
            "VIDEO_BYTETRACK_MATCHING_THRESHOLD", "0.90", 0.0, 1.0
        ),
        bytetrack_minimum_consecutive_frames=_positive_int(
            "VIDEO_BYTETRACK_MINIMUM_CONSECUTIVE_FRAMES", "1"
        ),
        scene_cut_enabled=_boolean("VIDEO_SCENE_CUT_ENABLED", "true"),
        scene_cut_histogram_correlation=_bounded_float(
            "VIDEO_SCENE_CUT_HISTOGRAM_CORRELATION", "0.35", -1.0, 1.0
        ),
        scene_cut_pixel_difference=_bounded_float(
            "VIDEO_SCENE_CUT_PIXEL_DIFFERENCE", "0.10", 0.0, 1.0
        ),
        appearance_enabled=_boolean("VIDEO_TRACK_APPEARANCE_ENABLED", "true"),
        appearance_min_similarity=_bounded_float(
            "VIDEO_TRACK_APPEARANCE_MIN_SIMILARITY", "0.20", -1.0, 1.0
        ),
        appearance_weight=_bounded_float(
            "VIDEO_TRACK_APPEARANCE_WEIGHT", "0.45", 0.0, 1.0
        ),
        appearance_momentum=_bounded_float(
            "VIDEO_TRACK_APPEARANCE_MOMENTUM", "0.80", 0.0, 1.0
        ),
    )


@lru_cache(maxsize=1)
def get_video_recognition_settings() -> VideoRecognitionSettings:
    return VideoRecognitionSettings(
        samples_per_track=_positive_int(
            "VIDEO_RECOGNITION_SAMPLES_PER_WINDOW",
            os.getenv("VIDEO_RECOGNITION_SAMPLES_PER_TRACK", "3"),
        ),
        anonymous_min_observations=_positive_int(
            "VIDEO_ANONYMOUS_MIN_OBSERVATIONS", "2"
        ),
        anonymous_min_quality_samples=_positive_int(
            "VIDEO_ANONYMOUS_MIN_QUALITY_SAMPLES", "3"
        ),
        anonymous_min_quality=_bounded_float(
            "VIDEO_ANONYMOUS_MIN_QUALITY", "0.50", 0.0, 1.0
        ),
        anonymous_min_frontal_score=_bounded_float(
            "VIDEO_ANONYMOUS_MIN_FRONTAL_SCORE", "0.35", 0.0, 1.0
        ),
        anonymous_min_duration_ms=round(
            _positive_float("VIDEO_ANONYMOUS_MIN_DURATION_SECONDS", "0.8")
            * 1000
        ),
        anonymous_min_detection_confidence=_bounded_float(
            "VIDEO_ANONYMOUS_MIN_DETECTION_CONFIDENCE", "0.50", 0.0, 1.0
        ),
        anonymous_min_face_size_px=_positive_int(
            "VIDEO_ANONYMOUS_MIN_FACE_SIZE_PX", "64"
        ),
        anonymous_min_sharpness=_positive_float(
            "VIDEO_ANONYMOUS_MIN_SHARPNESS", "35"
        ),
        anonymous_min_embedding_consistency=_bounded_float(
            "VIDEO_ANONYMOUS_MIN_EMBEDDING_CONSISTENCY", "0.35", 0.0, 1.0
        ),
        anonymous_single_min_quality=_bounded_float(
            "VIDEO_ANONYMOUS_SINGLE_MIN_QUALITY", "0.60", 0.0, 1.0
        ),
        anonymous_single_min_frontal_score=_bounded_float(
            "VIDEO_ANONYMOUS_SINGLE_MIN_FRONTAL_SCORE", "0.65", 0.0, 1.0
        ),
        anonymous_single_min_detection_confidence=_bounded_float(
            "VIDEO_ANONYMOUS_SINGLE_MIN_DETECTION_CONFIDENCE", "0.75", 0.0, 1.0
        ),
        anonymous_single_min_face_size_px=_positive_int(
            "VIDEO_ANONYMOUS_SINGLE_MIN_FACE_SIZE_PX", "80"
        ),
        anonymous_single_min_sharpness=_positive_float(
            "VIDEO_ANONYMOUS_SINGLE_MIN_SHARPNESS", "55"
        ),
        anonymous_short_min_observations=_positive_int(
            "VIDEO_ANONYMOUS_SHORT_MIN_OBSERVATIONS", "4"
        ),
        anonymous_short_min_duration_ms=round(
            _positive_float("VIDEO_ANONYMOUS_SHORT_MIN_DURATION_SECONDS", "0.5")
            * 1000
        ),
        window_ms=round(
            _positive_float("VIDEO_RECOGNITION_WINDOW_SECONDS", "3") * 1000
        ),
        strong_match_margin=_bounded_float(
            "VIDEO_RECOGNITION_STRONG_MATCH_MARGIN", "0.08", 0.0, 0.5
        ),
        weak_known_min_duration_ms=round(
            _positive_float(
                "VIDEO_RECOGNITION_WEAK_MATCH_MIN_DURATION_SECONDS",
                "1.0",
            )
            * 1000
        ),
        short_known_min_detection_confidence=_bounded_float(
            "VIDEO_RECOGNITION_SHORT_KNOWN_MIN_DETECTION_CONFIDENCE",
            "0.50",
            0.0,
            1.0,
        ),
        bridge_window_count=_positive_int(
            "VIDEO_RECOGNITION_BRIDGE_WINDOW_COUNT", "1"
        ),
        continuity_window_count=_positive_int(
            "VIDEO_RECOGNITION_CONTINUITY_WINDOWS", "4"
        ),
        continuity_threshold=_bounded_float(
            "VIDEO_RECOGNITION_CONTINUITY_THRESHOLD", "0.30", 0.0, 1.0
        ),
        continuity_distance_penalty=_bounded_float(
            "VIDEO_RECOGNITION_CONTINUITY_DISTANCE_PENALTY", "0.03", 0.0, 0.25
        ),
        reid_gallery_size=_positive_int("VIDEO_REID_GALLERY_SIZE", "5"),
        reid_min_similarity=_bounded_float(
            "VIDEO_REID_MIN_SIMILARITY", "0.30", 0.0, 1.0
        ),
    )
