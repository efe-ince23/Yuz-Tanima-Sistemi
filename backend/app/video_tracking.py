import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

from app.video_config import VideoTrackingSettings, get_video_tracking_settings
from app.video_detection import (
    DetectedVideoFrame,
    NormalizedBoundingBox,
    VideoDetectionSummary,
    VideoFaceDetection,
    detect_video_faces,
)


class VideoTrackingError(RuntimeError):
    pass


class SceneCutDetector:
    def __init__(self, settings: VideoTrackingSettings):
        self.settings = settings
        self._previous_histogram: Optional[np.ndarray] = None
        self._previous_gray: Optional[np.ndarray] = None

    def update(self, image: np.ndarray) -> bool:
        resized = cv2.resize(image, (64, 36), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
        cv2.normalize(histogram, histogram, alpha=1.0, norm_type=cv2.NORM_L1)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        is_cut = False
        if self._previous_histogram is not None and self._previous_gray is not None:
            correlation = cv2.compareHist(
                self._previous_histogram,
                histogram,
                cv2.HISTCMP_CORREL,
            )
            pixel_difference = float(
                np.mean(cv2.absdiff(self._previous_gray, gray)) / 255.0
            )
            is_cut = (
                correlation < self.settings.scene_cut_histogram_correlation
                and pixel_difference > self.settings.scene_cut_pixel_difference
            )
        self._previous_histogram = histogram
        self._previous_gray = gray
        return is_cut


@dataclass(frozen=True)
class TrackedVideoFace:
    track_id: int
    is_new_track: bool
    detection: VideoFaceDetection


@dataclass(frozen=True)
class TrackedVideoFrame:
    frame_number: int
    timestamp_ms: int
    width: int
    height: int
    image: np.ndarray
    faces: Tuple[TrackedVideoFace, ...]


@dataclass(frozen=True)
class VideoTrackSummary:
    track_id: int
    first_seen_ms: int
    last_seen_ms: int
    first_frame_number: int
    last_frame_number: int
    observation_count: int
    best_confidence: float
    best_frame_number: int


@dataclass(frozen=True)
class VideoTrackingSummary:
    detection: VideoDetectionSummary
    unique_track_count: int
    tracks: Tuple[VideoTrackSummary, ...]


@dataclass
class _TrackState:
    track_id: int
    first_seen_ms: int
    last_seen_ms: int
    first_frame_number: int
    last_frame_number: int
    observation_count: int
    best_confidence: float
    best_frame_number: int
    bounding_box: NormalizedBoundingBox
    previous_center: Optional[Tuple[float, float]] = None
    previous_timestamp_ms: Optional[int] = None
    appearance_embedding: Optional[np.ndarray] = None

    def predicted_center(self, timestamp_ms: int) -> Tuple[float, float]:
        current_center = _box_center(self.bounding_box)
        if self.previous_center is None or self.previous_timestamp_ms is None:
            return current_center
        elapsed = self.last_seen_ms - self.previous_timestamp_ms
        if elapsed <= 0:
            return current_center
        prediction_horizon = min(timestamp_ms - self.last_seen_ms, elapsed * 2)
        velocity_x = (current_center[0] - self.previous_center[0]) / elapsed
        velocity_y = (current_center[1] - self.previous_center[1]) / elapsed
        return (
            current_center[0] + velocity_x * prediction_horizon,
            current_center[1] + velocity_y * prediction_horizon,
        )

    def predicted_box(self, timestamp_ms: int) -> NormalizedBoundingBox:
        current_center = _box_center(self.bounding_box)
        predicted_center = self.predicted_center(timestamp_ms)
        shift_x = predicted_center[0] - current_center[0]
        shift_y = predicted_center[1] - current_center[1]
        return (
            self.bounding_box[0] + shift_x,
            self.bounding_box[1] + shift_y,
            self.bounding_box[2] + shift_x,
            self.bounding_box[3] + shift_y,
        )

    def update(
        self,
        frame: DetectedVideoFrame,
        face: VideoFaceDetection,
        appearance_momentum: float,
    ) -> None:
        self.previous_center = _box_center(self.bounding_box)
        self.previous_timestamp_ms = self.last_seen_ms
        self.bounding_box = face.normalized_bounding_box
        self.last_seen_ms = frame.timestamp_ms
        self.last_frame_number = frame.frame_number
        self.observation_count += 1
        if face.confidence > self.best_confidence:
            self.best_confidence = face.confidence
            self.best_frame_number = frame.frame_number
        embedding = _normalized_embedding(face.embedding)
        if embedding is not None:
            if self.appearance_embedding is None:
                self.appearance_embedding = embedding
            else:
                combined = (
                    appearance_momentum * self.appearance_embedding
                    + (1.0 - appearance_momentum) * embedding
                )
                self.appearance_embedding = _normalized_embedding(combined)


TrackedFrameHandler = Callable[[TrackedVideoFrame], None]


def _box_center(box: NormalizedBoundingBox) -> Tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _box_area(box: NormalizedBoundingBox) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _box_iou(first: NormalizedBoundingBox, second: NormalizedBoundingBox) -> float:
    intersection_width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    union = _box_area(first) + _box_area(second) - intersection
    return intersection / union if union > 0 else 0.0


def _normalized_embedding(embedding: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if embedding is None:
        return None
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-12:
        return None
    return vector / norm


class MultiFaceTracker:
    _INVALID_COST = 1_000_000.0

    def __init__(self, settings: Optional[VideoTrackingSettings] = None):
        self.settings = settings or get_video_tracking_settings()
        if self.settings.max_area_ratio < 1:
            raise ValueError("Takip alan orani en az 1 olmalidir.")
        self._tracks: Dict[int, _TrackState] = {}
        self._active_track_ids = set()
        self._next_track_id = 1
        self._last_frame_number: Optional[int] = None
        self._last_timestamp_ms: Optional[int] = None

    def _new_track(
        self,
        frame: DetectedVideoFrame,
        face: VideoFaceDetection,
    ) -> _TrackState:
        state = _TrackState(
            track_id=self._next_track_id,
            first_seen_ms=frame.timestamp_ms,
            last_seen_ms=frame.timestamp_ms,
            first_frame_number=frame.frame_number,
            last_frame_number=frame.frame_number,
            observation_count=1,
            best_confidence=face.confidence,
            best_frame_number=frame.frame_number,
            bounding_box=face.normalized_bounding_box,
            appearance_embedding=_normalized_embedding(face.embedding),
        )
        self._tracks[state.track_id] = state
        self._active_track_ids.add(state.track_id)
        self._next_track_id += 1
        return state

    def reset_active_tracks(self) -> None:
        self._active_track_ids.clear()

    def _pair_cost(
        self,
        track: _TrackState,
        face: VideoFaceDetection,
        timestamp_ms: int,
    ) -> float:
        box = face.normalized_bounding_box
        predicted_box = track.predicted_box(timestamp_ms)
        iou = _box_iou(predicted_box, box)
        predicted_center = track.predicted_center(timestamp_ms)
        face_center = _box_center(box)
        center_distance = math.dist(predicted_center, face_center) / math.sqrt(2)
        previous_area = _box_area(track.bounding_box)
        current_area = _box_area(box)
        if previous_area <= 0 or current_area <= 0:
            return self._INVALID_COST
        area_ratio = max(previous_area, current_area) / min(previous_area, current_area)
        if area_ratio > self.settings.max_area_ratio:
            return self._INVALID_COST
        if iou < self.settings.min_iou and center_distance > self.settings.max_center_distance:
            return self._INVALID_COST
        area_penalty = min(
            math.log(area_ratio) / math.log(self.settings.max_area_ratio)
            if self.settings.max_area_ratio > 1
            else 0.0,
            1.0,
        )
        geometry_cost = (
            0.55 * (1.0 - iou)
            + 0.35 * center_distance
            + 0.10 * area_penalty
        )
        face_embedding = _normalized_embedding(face.embedding)
        if (
            not self.settings.appearance_enabled
            or track.appearance_embedding is None
            or face_embedding is None
        ):
            return geometry_cost
        similarity = float(np.dot(track.appearance_embedding, face_embedding))
        if similarity < self.settings.appearance_min_similarity:
            return self._INVALID_COST
        appearance_cost = 1.0 - max(0.0, min(1.0, similarity))
        weight = self.settings.appearance_weight
        return (1.0 - weight) * geometry_cost + weight * appearance_cost

    def update(self, frame: DetectedVideoFrame) -> TrackedVideoFrame:
        if self._last_frame_number is not None and frame.frame_number <= self._last_frame_number:
            raise VideoTrackingError("Video kareleri artan sirada takip edilmelidir.")
        if self._last_timestamp_ms is not None and frame.timestamp_ms < self._last_timestamp_ms:
            raise VideoTrackingError("Video zaman damgalari geriye gidemez.")
        self._last_frame_number = frame.frame_number
        self._last_timestamp_ms = frame.timestamp_ms

        active_tracks = [
            track
            for track in self._tracks.values()
            if track.track_id in self._active_track_ids
            and frame.timestamp_ms - track.last_seen_ms <= self.settings.max_gap_ms
        ]
        self._active_track_ids = {track.track_id for track in active_tracks}
        assignments: Dict[int, int] = {}
        if active_tracks and frame.faces:
            costs = [
                [self._pair_cost(track, face, frame.timestamp_ms) for face in frame.faces]
                for track in active_tracks
            ]
            row_indexes, column_indexes = linear_sum_assignment(costs)
            for row_index, column_index in zip(row_indexes, column_indexes):
                if costs[row_index][column_index] < self._INVALID_COST:
                    assignments[column_index] = active_tracks[row_index].track_id

        tracked_faces: List[TrackedVideoFace] = []
        for face_index, face in enumerate(frame.faces):
            track_id = assignments.get(face_index)
            is_new_track = track_id is None
            if track_id is None:
                track = self._new_track(frame, face)
            else:
                track = self._tracks[track_id]
                track.update(frame, face, self.settings.appearance_momentum)
            tracked_faces.append(
                TrackedVideoFace(
                    track_id=track.track_id,
                    is_new_track=is_new_track,
                    detection=face,
                )
            )

        return TrackedVideoFrame(
            frame_number=frame.frame_number,
            timestamp_ms=frame.timestamp_ms,
            width=frame.width,
            height=frame.height,
            image=frame.image,
            faces=tuple(tracked_faces),
        )

    def summaries(self) -> Tuple[VideoTrackSummary, ...]:
        return tuple(
            VideoTrackSummary(
                track_id=track.track_id,
                first_seen_ms=track.first_seen_ms,
                last_seen_ms=track.last_seen_ms,
                first_frame_number=track.first_frame_number,
                last_frame_number=track.last_frame_number,
                observation_count=track.observation_count,
                best_confidence=track.best_confidence,
                best_frame_number=track.best_frame_number,
            )
            for track in sorted(self._tracks.values(), key=lambda item: item.track_id)
        )


class ByteTrackFaceTracker:
    def __init__(self, settings: VideoTrackingSettings, sample_fps: float):
        try:
            import supervision as sv
        except ImportError as error:
            raise VideoTrackingError(
                "ByteTrack icin supervision paketi kurulu degil."
            ) from error

        self._sv = sv
        self._settings = settings
        self._sample_fps = sample_fps
        self._tracker = self._create_tracker()
        self._raw_to_global: Dict[int, int] = {}
        self._next_track_id = 1
        self._tracks: Dict[int, VideoTrackSummary] = {}
        self._last_frame_number: Optional[int] = None
        self._last_timestamp_ms: Optional[int] = None

    def _create_tracker(self):
        lost_track_buffer = max(
            1, round(self._settings.max_gap_ms / 1000.0 * self._sample_fps)
        )
        return self._sv.ByteTrack(
            track_activation_threshold=self._settings.bytetrack_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=self._settings.bytetrack_matching_threshold,
            frame_rate=max(1, round(self._sample_fps)),
            minimum_consecutive_frames=(
                self._settings.bytetrack_minimum_consecutive_frames
            ),
        )

    def reset_active_tracks(self) -> None:
        self._tracker = self._create_tracker()
        self._raw_to_global.clear()

    def update(self, frame: DetectedVideoFrame) -> TrackedVideoFrame:
        if (
            self._last_frame_number is not None
            and frame.frame_number <= self._last_frame_number
        ):
            raise VideoTrackingError("Video kareleri artan sirada takip edilmelidir.")
        if (
            self._last_timestamp_ms is not None
            and frame.timestamp_ms < self._last_timestamp_ms
        ):
            raise VideoTrackingError("Video zaman damgalari geriye gidemez.")
        self._last_frame_number = frame.frame_number
        self._last_timestamp_ms = frame.timestamp_ms

        if frame.faces:
            boxes = np.asarray(
                [face.bounding_box for face in frame.faces],
                dtype=np.float32,
            )
            confidences = np.asarray(
                [face.confidence for face in frame.faces],
                dtype=np.float32,
            )
            class_ids = np.zeros(len(frame.faces), dtype=int)
            source_indexes = np.arange(len(frame.faces), dtype=int)
        else:
            boxes = np.empty((0, 4), dtype=np.float32)
            confidences = np.empty((0,), dtype=np.float32)
            class_ids = np.empty((0,), dtype=int)
            source_indexes = np.empty((0,), dtype=int)

        detections = self._sv.Detections(
            xyxy=boxes,
            confidence=confidences,
            class_id=class_ids,
            data={"source_index": source_indexes},
        )
        tracked = self._tracker.update_with_detections(detections)
        tracked_faces: List[TrackedVideoFace] = []
        tracker_ids = tracked.tracker_id
        tracked_source_indexes = tracked.data.get("source_index", [])
        if tracker_ids is not None:
            for source_index, raw_track_id in zip(
                tracked_source_indexes,
                tracker_ids,
            ):
                raw_track_id = int(raw_track_id)
                track_id = self._raw_to_global.get(raw_track_id)
                if track_id is None:
                    track_id = self._next_track_id
                    self._next_track_id += 1
                    self._raw_to_global[raw_track_id] = track_id
                face = frame.faces[int(source_index)]
                previous = self._tracks.get(track_id)
                is_new_track = previous is None
                if previous is None:
                    summary = VideoTrackSummary(
                        track_id=track_id,
                        first_seen_ms=frame.timestamp_ms,
                        last_seen_ms=frame.timestamp_ms,
                        first_frame_number=frame.frame_number,
                        last_frame_number=frame.frame_number,
                        observation_count=1,
                        best_confidence=face.confidence,
                        best_frame_number=frame.frame_number,
                    )
                else:
                    best_confidence = previous.best_confidence
                    best_frame_number = previous.best_frame_number
                    if face.confidence > best_confidence:
                        best_confidence = face.confidence
                        best_frame_number = frame.frame_number
                    summary = VideoTrackSummary(
                        track_id=track_id,
                        first_seen_ms=previous.first_seen_ms,
                        last_seen_ms=frame.timestamp_ms,
                        first_frame_number=previous.first_frame_number,
                        last_frame_number=frame.frame_number,
                        observation_count=previous.observation_count + 1,
                        best_confidence=best_confidence,
                        best_frame_number=best_frame_number,
                    )
                self._tracks[track_id] = summary
                tracked_faces.append(
                    TrackedVideoFace(
                        track_id=track_id,
                        is_new_track=is_new_track,
                        detection=face,
                    )
                )

        return TrackedVideoFrame(
            frame_number=frame.frame_number,
            timestamp_ms=frame.timestamp_ms,
            width=frame.width,
            height=frame.height,
            image=frame.image,
            faces=tuple(tracked_faces),
        )

    def summaries(self) -> Tuple[VideoTrackSummary, ...]:
        return tuple(
            self._tracks[track_id]
            for track_id in sorted(self._tracks)
        )


def track_video_faces(
    object_path: str,
    sample_fps: float,
    frame_handler: Optional[TrackedFrameHandler] = None,
    settings: Optional[VideoTrackingSettings] = None,
) -> VideoTrackingSummary:
    effective_settings = settings or get_video_tracking_settings()
    if effective_settings.engine == "bytetrack":
        tracker = ByteTrackFaceTracker(effective_settings, sample_fps)
    elif effective_settings.engine == "custom":
        tracker = MultiFaceTracker(effective_settings)
    else:
        raise VideoTrackingError(
            f"Desteklenmeyen video tracker: {effective_settings.engine}"
        )
    scene_detector = SceneCutDetector(effective_settings)

    def handle_detected_frame(frame: DetectedVideoFrame) -> None:
        if effective_settings.scene_cut_enabled and scene_detector.update(frame.image):
            tracker.reset_active_tracks()
        tracked_frame = tracker.update(frame)
        if frame_handler is not None:
            frame_handler(tracked_frame)

    detection = detect_video_faces(
        object_path,
        sample_fps,
        frame_handler=handle_detected_frame,
    )
    tracks = tracker.summaries()
    return VideoTrackingSummary(
        detection=detection,
        unique_track_count=len(tracks),
        tracks=tracks,
    )
