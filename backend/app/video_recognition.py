from dataclasses import dataclass
import time
from typing import Callable, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

import cv2
import numpy as np
from sqlalchemy.orm import Session

from app.face_detector import ANONYMOUS_MATCH_THRESHOLD, MATCH_THRESHOLD
from app.face_identification import (
    create_anonymous_identity,
    find_closest_anonymous_face,
    find_closest_face,
    lock_anonymous_matching,
    record_anonymous_observation,
)
from app.face_storage import (
    delete_face_image,
    save_anonymous_face_crop,
)
from app.vector_store import synchronize_face_id_safely
from app.video_config import get_video_recognition_settings
from app.video_tracking import (
    TrackedVideoFace,
    TrackedVideoFrame,
    VideoTrackingSummary,
    track_video_faces,
)
from app.yolo_arcface import DetectedYoloFace, get_yolo_arcface_engine


@dataclass(frozen=True)
class RecognizedVideoTrack:
    track_id: int
    face_id: UUID
    status: str
    recognized: bool
    similarity: Optional[float]
    threshold: float
    person_id: Optional[int]
    name: Optional[str]
    metadata: Optional[Dict[str, object]]
    matched_image_path: Optional[str]
    representative_frame_number: int
    representative_timestamp_ms: int
    detection_confidence: float
    source_track_id: Optional[int] = None
    first_seen_ms: Optional[int] = None
    last_seen_ms: Optional[int] = None
    observation_count: Optional[int] = None


@dataclass(frozen=True)
class VideoRecognitionSummary:
    tracking: VideoTrackingSummary
    tracks: Tuple[RecognizedVideoTrack, ...]
    changed_face_ids: Tuple[UUID, ...] = ()
    stored_anonymous_image_paths: Tuple[str, ...] = ()
    tracking_seconds: float = 0.0
    identity_seconds: float = 0.0


@dataclass
class _TrackRepresentative:
    track_id: int
    frame_number: int
    timestamp_ms: int
    detection_confidence: float
    quality: float
    frontal_score: float
    face_min_size_px: int
    sharpness: float
    aligned_face: np.ndarray
    face_crop: np.ndarray
    embedding: Optional[np.ndarray] = None


@dataclass
class _TrackWindow:
    source_track_id: int
    window_index: int
    first_seen_ms: int
    last_seen_ms: int
    observation_count: int
    best_detection_confidence: float
    samples: List[_TrackRepresentative]


@dataclass
class _WindowDecision:
    window: _TrackWindow
    face_id: Optional[UUID]
    known_votes: List[Tuple[object, object, float, _TrackRepresentative]]
    embeddings: List[np.ndarray]


@dataclass
class _InVideoIdentityGallery:
    identity: object
    embeddings: List[np.ndarray]


@dataclass
class _ReadOnlyAnonymousIdentity:
    face_id: UUID


RecognitionFrameHandler = Callable[[TrackedVideoFrame], None]


def _frontal_score(landmarks: np.ndarray) -> float:
    if landmarks.shape != (5, 2):
        return 0.5
    left_eye, right_eye, nose, left_mouth, right_mouth = landmarks
    eye_axis = right_eye - left_eye
    eye_distance = float(np.linalg.norm(eye_axis))
    if eye_distance <= 1e-6:
        return 0.0
    horizontal_axis = eye_axis / eye_distance
    eye_midpoint = (left_eye + right_eye) / 2.0
    mouth_midpoint = (left_mouth + right_mouth) / 2.0

    # Project offsets onto the eye line. This measures yaw while remaining
    # invariant to an otherwise frontal face being tilted in the image.
    nose_offset = abs(float(np.dot(nose - eye_midpoint, horizontal_axis))) / eye_distance
    mouth_offset = abs(
        float(np.dot(mouth_midpoint - eye_midpoint, horizontal_axis))
    ) / eye_distance
    nose_symmetry = 1.0 - min(1.0, nose_offset / 0.45)
    mouth_symmetry = 1.0 - min(1.0, mouth_offset / 0.60)
    score = 0.75 * nose_symmetry + 0.25 * mouth_symmetry
    return max(0.0, min(1.0, score))


def _expanded_face_crop(
    frame: TrackedVideoFrame,
    tracked_face: TrackedVideoFace,
) -> np.ndarray:
    x1, y1, x2, y2 = tracked_face.detection.bounding_box
    padding = max(8, round(max(x2 - x1, y2 - y1) * 0.15))
    crop_x1 = max(0, x1 - padding)
    crop_y1 = max(0, y1 - padding)
    crop_x2 = min(frame.width, x2 + padding)
    crop_y2 = min(frame.height, y2 + padding)
    return frame.image[crop_y1:crop_y2, crop_x1:crop_x2].copy()


def _representative_quality(
    frame: TrackedVideoFrame,
    tracked_face: TrackedVideoFace,
) -> float:
    x1, y1, x2, y2 = tracked_face.detection.normalized_bounding_box
    normalized_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    size_score = min(1.0, normalized_area / 0.08)

    crop = _expanded_face_crop(frame, tracked_face)
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharpness_score = min(1.0, sharpness / 180.0)
    brightness = float(np.mean(gray))
    lighting_score = max(0.0, 1.0 - abs(brightness - 128.0) / 128.0)

    landmarks = np.asarray(tracked_face.detection.landmarks, dtype=np.float32)
    frontal_score = _frontal_score(landmarks)

    return float(
        0.30 * tracked_face.detection.confidence
        + 0.20 * size_score
        + 0.25 * sharpness_score
        + 0.10 * lighting_score
        + 0.15 * frontal_score
    )


def _collect_representative(
    windows: Dict[Tuple[int, int], _TrackWindow],
    frame: TrackedVideoFrame,
    tracked_face: TrackedVideoFace,
    maximum_samples: int,
    window_ms: int,
) -> None:
    quality = _representative_quality(frame, tracked_face)
    window_index = frame.timestamp_ms // window_ms
    window_key = (tracked_face.track_id, window_index)
    window = windows.get(window_key)
    if window is None:
        window = _TrackWindow(
            source_track_id=tracked_face.track_id,
            window_index=window_index,
            first_seen_ms=frame.timestamp_ms,
            last_seen_ms=frame.timestamp_ms,
            observation_count=0,
            best_detection_confidence=tracked_face.detection.confidence,
            samples=[],
        )
        windows[window_key] = window
    window.first_seen_ms = min(window.first_seen_ms, frame.timestamp_ms)
    window.last_seen_ms = max(window.last_seen_ms, frame.timestamp_ms)
    window.observation_count += 1
    window.best_detection_confidence = max(
        window.best_detection_confidence,
        tracked_face.detection.confidence,
    )
    if len(window.samples) >= maximum_samples and quality <= window.samples[-1].quality:
        return

    detection = tracked_face.detection
    yolo_face = DetectedYoloFace(
        bbox=np.asarray(detection.bounding_box, dtype=np.float32),
        landmarks=np.asarray(detection.landmarks, dtype=np.float32),
        confidence=detection.confidence,
    )
    aligned_face = get_yolo_arcface_engine().recognizer.align(frame.image, yolo_face)
    face_crop = _expanded_face_crop(frame, tracked_face)
    if face_crop.size == 0:
        raise RuntimeError("Video yuz ornegi kirpilamadi.")
    gray_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    x1, y1, x2, y2 = detection.bounding_box
    representative = _TrackRepresentative(
        track_id=tracked_face.track_id,
        frame_number=frame.frame_number,
        timestamp_ms=frame.timestamp_ms,
        detection_confidence=detection.confidence,
        quality=quality,
        frontal_score=_frontal_score(
            np.asarray(detection.landmarks, dtype=np.float32)
        ),
        face_min_size_px=max(0, min(x2 - x1, y2 - y1)),
        sharpness=float(cv2.Laplacian(gray_crop, cv2.CV_64F).var()),
        aligned_face=aligned_face,
        face_crop=face_crop,
        embedding=(
            detection.embedding.copy()
            if detection.embedding is not None
            else None
        ),
    )
    window.samples.append(representative)
    window.samples.sort(key=lambda item: (-item.quality, item.frame_number))
    del window.samples[maximum_samples:]


def _normalized_average(embeddings: List[np.ndarray]) -> np.ndarray:
    combined = np.sum(embeddings, axis=0)
    norm = float(np.linalg.norm(combined))
    return combined / max(norm, 1e-12)


def _is_standard_anonymous_sample(representative, settings) -> bool:
    return (
        representative.quality >= settings.anonymous_min_quality
        and representative.frontal_score >= settings.anonymous_min_frontal_score
        and representative.detection_confidence
        >= settings.anonymous_min_detection_confidence
        and representative.face_min_size_px >= settings.anonymous_min_face_size_px
        and representative.sharpness >= settings.anonymous_min_sharpness
    )


def _is_exceptional_single_anonymous_sample(representative, settings) -> bool:
    return (
        representative.quality >= settings.anonymous_single_min_quality
        and representative.frontal_score
        >= settings.anonymous_single_min_frontal_score
        and representative.detection_confidence
        >= settings.anonymous_single_min_detection_confidence
        and representative.face_min_size_px
        >= settings.anonymous_single_min_face_size_px
        and representative.sharpness >= settings.anonymous_single_min_sharpness
    )


def _largest_consistent_sample_group(
    samples: List[Tuple[_TrackRepresentative, np.ndarray]],
    minimum_similarity: float,
) -> List[Tuple[_TrackRepresentative, np.ndarray]]:
    if not samples:
        return []
    remaining = set(range(len(samples)))
    groups: List[List[int]] = []
    while remaining:
        pending = [remaining.pop()]
        group = []
        while pending:
            current = pending.pop()
            group.append(current)
            connected = [
                candidate
                for candidate in remaining
                if float(np.dot(samples[current][1], samples[candidate][1]))
                >= minimum_similarity
            ]
            for candidate in connected:
                remaining.remove(candidate)
                pending.append(candidate)
        groups.append(group)
    selected = max(
        groups,
        key=lambda group: (
            len(group),
            sum(samples[index][0].quality for index in group),
        ),
    )
    return [samples[index] for index in selected]


def _same_window_identity(first: _WindowDecision, second: _WindowDecision) -> bool:
    if first.face_id is not None or second.face_id is not None:
        return first.face_id is not None and first.face_id == second.face_id
    first_embedding = _normalized_average(first.embeddings)
    second_embedding = _normalized_average(second.embeddings)
    return float(np.dot(first_embedding, second_embedding)) >= ANONYMOUS_MATCH_THRESHOLD


def _bridge_known_identity_gaps(
    decisions: List[_WindowDecision],
    maximum_gap_windows: int,
) -> None:
    if maximum_gap_windows < 1:
        return
    by_track: Dict[int, List[_WindowDecision]] = {}
    for decision in decisions:
        by_track.setdefault(decision.window.source_track_id, []).append(decision)

    for track_decisions in by_track.values():
        for start_index, first in enumerate(track_decisions):
            if first.face_id is None:
                continue
            for gap_size in range(1, maximum_gap_windows + 1):
                end_index = start_index + gap_size + 1
                if end_index >= len(track_decisions):
                    break
                middle = track_decisions[start_index + 1 : end_index]
                last = track_decisions[end_index]
                consecutive = all(
                    current.window.window_index
                    == previous.window.window_index + 1
                    for previous, current in zip(
                        track_decisions[start_index:end_index],
                        track_decisions[start_index + 1 : end_index + 1],
                    )
                )
                if (
                    consecutive
                    and all(item.face_id is None for item in middle)
                    and last.face_id == first.face_id
                ):
                    for item in middle:
                        item.face_id = first.face_id
                    break


def _stabilize_known_identity_continuity(
    decisions: List[_WindowDecision],
    maximum_distance_windows: int,
    minimum_similarity: float,
    distance_penalty: float,
) -> None:
    by_track: Dict[int, List[_WindowDecision]] = {}
    for decision in decisions:
        by_track.setdefault(decision.window.source_track_id, []).append(decision)

    for track_decisions in by_track.values():
        anchors = [
            decision
            for decision in track_decisions
            if decision.face_id is not None and decision.known_votes
        ]
        for decision in track_decisions:
            if decision.face_id is not None:
                continue
            candidates = []
            for anchor in anchors:
                window_distance = abs(
                    decision.window.window_index - anchor.window.window_index
                )
                if window_distance < 1 or window_distance > maximum_distance_windows:
                    continue
                similarity, representative = max(
                    (
                        (float(np.dot(candidate, reference)), representative)
                        for candidate, representative in zip(
                            decision.embeddings,
                            decision.window.samples,
                        )
                        for reference in anchor.embeddings
                    ),
                    key=lambda item: item[0],
                )
                required_similarity = min(
                    1.0,
                    minimum_similarity
                    + distance_penalty * max(0, window_distance - 1),
                )
                if similarity >= required_similarity:
                    anchor_vote = max(anchor.known_votes, key=lambda vote: vote[2])
                    candidates.append(
                        (
                            similarity,
                            anchor.face_id,
                            anchor_vote[0],
                            anchor_vote[1],
                            representative,
                        )
                    )
            if not candidates:
                continue
            candidates.sort(key=lambda item: item[0], reverse=True)
            if (
                len(candidates) > 1
                and candidates[0][1] != candidates[1][1]
                and candidates[0][0] - candidates[1][0] < 0.05
            ):
                continue
            similarity, face_id, person, face_image, representative = candidates[0]
            decision.face_id = face_id
            decision.known_votes = [
                (person, face_image, similarity, representative)
            ]


def _propagate_known_identity_across_tracks(
    decisions: List[_WindowDecision],
    minimum_similarity: float,
) -> None:
    anchors = [
        decision
        for decision in decisions
        if decision.face_id is not None and decision.known_votes
    ]
    for decision in decisions:
        if decision.face_id is not None:
            continue
        candidates = []
        for anchor in anchors:
            if anchor.window.window_index != decision.window.window_index:
                continue
            best_similarity = -1.0
            best_representative = None
            for representative, embedding in zip(
                decision.window.samples,
                decision.embeddings,
            ):
                similarity = max(
                    float(np.dot(embedding, reference))
                    for reference in anchor.embeddings
                )
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_representative = representative
            if best_similarity < minimum_similarity or best_representative is None:
                continue
            anchor_vote = max(anchor.known_votes, key=lambda vote: vote[2])
            candidates.append(
                (
                    best_similarity,
                    anchor.face_id,
                    anchor_vote[0],
                    anchor_vote[1],
                    best_representative,
                )
            )
        if not candidates:
            continue
        candidates.sort(key=lambda item: item[0], reverse=True)
        if (
            len(candidates) > 1
            and candidates[0][1] != candidates[1][1]
            and candidates[0][0] - candidates[1][0] < 0.05
        ):
            continue
        similarity, face_id, person, face_image, representative = candidates[0]
        decision.face_id = face_id
        decision.known_votes = [
            (person, face_image, similarity, representative)
        ]


def _gallery_similarity(
    embedding: np.ndarray,
    gallery: _InVideoIdentityGallery,
) -> float:
    return max(float(np.dot(embedding, candidate)) for candidate in gallery.embeddings)


def _update_gallery(
    gallery: _InVideoIdentityGallery,
    embedding: np.ndarray,
    maximum_size: int,
) -> None:
    if any(float(np.dot(embedding, item)) >= 0.995 for item in gallery.embeddings):
        return
    gallery.embeddings.append(embedding.copy())
    if len(gallery.embeddings) > maximum_size:
        gallery.embeddings.pop(0)


def recognize_video_tracks(
    session: Session,
    object_path: str,
    sample_fps: float,
    frame_handler: Optional[RecognitionFrameHandler] = None,
    *,
    manage_transaction: bool = True,
    read_only: bool = False,
    owner_user_id: Optional[UUID] = None,
) -> VideoRecognitionSummary:
    settings = get_video_recognition_settings()
    windows: Dict[Tuple[int, int], _TrackWindow] = {}

    def handle_tracked_frame(frame: TrackedVideoFrame) -> None:
        for tracked_face in frame.faces:
            _collect_representative(
                windows,
                frame,
                tracked_face,
                settings.samples_per_track,
                settings.window_ms,
            )
        if frame_handler is not None:
            frame_handler(frame)

    tracking_started = time.perf_counter()
    tracking = track_video_faces(
        object_path,
        sample_fps,
        frame_handler=handle_tracked_frame,
    )
    tracking_seconds = time.perf_counter() - tracking_started
    identity_started = time.perf_counter()
    ordered_windows = [windows[key] for key in sorted(windows)]
    flattened_representatives = [
        representative
        for window in ordered_windows
        for representative in window.samples
    ]
    if not flattened_representatives:
        return VideoRecognitionSummary(
            tracking=tracking,
            tracks=(),
            tracking_seconds=tracking_seconds,
            identity_seconds=time.perf_counter() - identity_started,
        )

    missing_representatives = [
        representative
        for representative in flattened_representatives
        if representative.embedding is None
    ]
    if missing_representatives:
        generated_embeddings = get_yolo_arcface_engine().recognizer.embed_aligned(
            [
                representative.aligned_face
                for representative in missing_representatives
            ]
        )
        if len(generated_embeddings) != len(missing_representatives):
            raise RuntimeError("ArcFace tum video izleri icin vektor uretemedi.")
        for representative, embedding in zip(
            missing_representatives,
            generated_embeddings,
        ):
            representative.embedding = embedding

    embeddings = [
        representative.embedding
        for representative in flattened_representatives
    ]
    if any(embedding is None for embedding in embeddings):
        raise RuntimeError("ArcFace video yuz vektoru eksik kaldi.")

    embeddings_by_window: Dict[Tuple[int, int], List[np.ndarray]] = {}
    embedding_index = 0
    for window in ordered_windows:
        sample_count = len(window.samples)
        embeddings_by_window[(window.source_track_id, window.window_index)] = list(
            embeddings[embedding_index : embedding_index + sample_count]
        )
        embedding_index += sample_count

    window_decisions: List[_WindowDecision] = []
    for window in ordered_windows:
        window_embeddings = embeddings_by_window[
            (window.source_track_id, window.window_index)
        ]
        known_votes: Dict[
            UUID,
            List[Tuple[object, object, float, _TrackRepresentative]],
        ] = {}
        for representative, embedding in zip(window.samples, window_embeddings):
            known_match = find_closest_face(
                session, embedding.tolist(), owner_user_id=owner_user_id
            )
            if known_match is None or known_match[2] < MATCH_THRESHOLD:
                continue
            person, face_image, similarity = known_match
            known_votes.setdefault(person.face_id, []).append(
                (person, face_image, float(similarity), representative)
            )
        required_votes = len(window.samples) // 2 + 1
        winning_face_id = max(
            known_votes,
            key=lambda face_id: (
                sum(
                    vote[2] * vote[3].quality
                    for vote in known_votes[face_id]
                ),
                len(known_votes[face_id]),
                sum(vote[2] for vote in known_votes[face_id])
                / len(known_votes[face_id]),
            ),
            default=None,
        )
        winning_votes = known_votes.get(winning_face_id, [])
        strong_single_vote = (
            len(winning_votes) == 1
            and winning_votes[0][2]
            >= MATCH_THRESHOLD + settings.strong_match_margin
            and winning_votes[0][3].quality >= 0.55
        )
        accepted = len(winning_votes) >= required_votes or strong_single_vote
        window_decisions.append(
            _WindowDecision(
                window=window,
                face_id=(
                    winning_face_id
                    if accepted
                    else None
                ),
                known_votes=(
                    winning_votes
                    if accepted
                    else []
                ),
                embeddings=window_embeddings,
            )
        )

    _propagate_known_identity_across_tracks(
        window_decisions,
        MATCH_THRESHOLD + settings.strong_match_margin,
    )
    _bridge_known_identity_gaps(
        window_decisions,
        settings.bridge_window_count,
    )
    _stabilize_known_identity_continuity(
        window_decisions,
        settings.continuity_window_count,
        settings.continuity_threshold,
        settings.continuity_distance_penalty,
    )

    decision_segments: List[List[_WindowDecision]] = []
    for decision in window_decisions:
        can_extend = (
            bool(decision_segments)
            and decision_segments[-1][-1].window.source_track_id
            == decision.window.source_track_id
            and _same_window_identity(decision_segments[-1][-1], decision)
        )
        if can_extend:
            decision_segments[-1].append(decision)
        else:
            decision_segments.append([decision])

    results: List[RecognizedVideoTrack] = []
    stored_anonymous_images: List[str] = []
    synchronized_face_ids = set()
    in_video_galleries: List[_InVideoIdentityGallery] = []
    anonymous_changed = False

    try:
        if not read_only:
            lock_anonymous_matching(session)
        for result_track_id, segment in enumerate(decision_segments, start=1):
            source_track_id = segment[0].window.source_track_id
            first_seen_ms = min(item.window.first_seen_ms for item in segment)
            last_seen_ms = max(item.window.last_seen_ms for item in segment)
            observation_count = sum(
                item.window.observation_count for item in segment
            )
            best_detection_confidence = max(
                item.window.best_detection_confidence for item in segment
            )
            segment_representatives = [
                representative
                for item in segment
                for representative in item.window.samples
            ]
            representative = max(
                segment_representatives,
                key=lambda item: (item.quality, -item.frame_number),
            )

            known_face_id = segment[0].face_id
            if known_face_id is not None:
                winning_votes = [
                    vote
                    for item in segment
                    for vote in item.known_votes
                    if vote[0].face_id == known_face_id
                ]
                if not winning_votes:
                    known_face_id = None

            if known_face_id is not None:
                best_similarity = max(vote[2] for vote in winning_votes)
                segment_duration_ms = max(0, last_seen_ms - first_seen_ms)
                known_detection_confidence = max(
                    vote[3].detection_confidence for vote in winning_votes
                )
                if (
                    best_similarity
                    < MATCH_THRESHOLD + settings.strong_match_margin
                    and segment_duration_ms < settings.weak_known_min_duration_ms
                ):
                    continue
                if (
                    segment_duration_ms < settings.weak_known_min_duration_ms
                    and known_detection_confidence
                    < settings.short_known_min_detection_confidence
                ):
                    continue
                person = winning_votes[0][0]
                average_similarity = sum(vote[2] for vote in winning_votes) / len(
                    winning_votes
                )
                best_vote = max(winning_votes, key=lambda vote: vote[2])
                face_image = best_vote[1]
                representative = best_vote[3]
                results.append(
                    RecognizedVideoTrack(
                        track_id=result_track_id,
                        face_id=person.face_id,
                        status="known",
                        recognized=True,
                        similarity=round(float(average_similarity), 4),
                        threshold=MATCH_THRESHOLD,
                        person_id=person.id,
                        name=f"{person.first_name} {person.last_name}",
                        metadata={"description": person.description},
                        matched_image_path=(
                            face_image.image_path if face_image is not None else None
                        ),
                        representative_frame_number=representative.frame_number,
                        representative_timestamp_ms=representative.timestamp_ms,
                        detection_confidence=round(
                            representative.detection_confidence, 4
                        ),
                        source_track_id=source_track_id,
                        first_seen_ms=first_seen_ms,
                        last_seen_ms=last_seen_ms,
                        observation_count=observation_count,
                    )
                )
                continue

            segment_duration_ms = max(0, last_seen_ms - first_seen_ms)
            candidate_samples = [
                (representative, embedding)
                for item in segment
                for representative, embedding in zip(
                    item.window.samples,
                    item.embeddings,
                )
            ]
            standard_samples = _largest_consistent_sample_group(
                [
                    item
                    for item in candidate_samples
                    if _is_standard_anonymous_sample(item[0], settings)
                ],
                settings.anonymous_min_embedding_consistency,
            )
            standard_track_is_valid = (
                observation_count >= settings.anonymous_min_observations
                and segment_duration_ms >= settings.anonymous_min_duration_ms
                and len(standard_samples) >= settings.anonymous_min_quality_samples
            )
            if standard_track_is_valid:
                eligible_samples = standard_samples
            else:
                exceptional_track_is_valid = (
                    observation_count == 1
                    or (
                        observation_count
                        >= settings.anonymous_short_min_observations
                        and segment_duration_ms
                        >= settings.anonymous_short_min_duration_ms
                    )
                )
                if not exceptional_track_is_valid:
                    continue
                exceptional_candidates = [
                    item
                    for item in candidate_samples
                    if _is_exceptional_single_anonymous_sample(item[0], settings)
                ]
                exceptional_samples = _largest_consistent_sample_group(
                    exceptional_candidates,
                    settings.anonymous_min_embedding_consistency,
                )
                required_exceptional_samples = len(candidate_samples) // 2 + 1
                if len(exceptional_samples) < required_exceptional_samples:
                    continue
                eligible_samples = exceptional_samples
            representative = max(
                (item[0] for item in eligible_samples),
                key=lambda item: (item.quality, -item.frame_number),
            )
            embedding = _normalized_average([item[1] for item in eligible_samples])
            embedding_list = embedding.tolist()

            gallery_identity = None
            gallery_similarity = None
            best_gallery = None
            for gallery in in_video_galleries:
                candidate_similarity = _gallery_similarity(embedding, gallery)
                if (
                    gallery_similarity is None
                    or candidate_similarity > gallery_similarity
                ):
                    gallery_identity = gallery.identity
                    gallery_similarity = candidate_similarity
                    best_gallery = gallery

            anonymous_match = find_closest_anonymous_face(
                session, embedding_list, owner_user_id=owner_user_id
            )
            database_match_is_valid = (
                anonymous_match is not None
                and anonymous_match[1] >= ANONYMOUS_MATCH_THRESHOLD
            )
            gallery_match_is_valid = (
                gallery_identity is not None
                and gallery_similarity is not None
                and gallery_similarity >= settings.reid_min_similarity
            )
            anonymous_identity = None
            anonymous_similarity = None
            matched_gallery = None
            if (
                gallery_match_is_valid
            ):
                anonymous_identity = gallery_identity
                anonymous_similarity = gallery_similarity
                matched_gallery = best_gallery
            elif database_match_is_valid:
                anonymous_identity, anonymous_similarity = anonymous_match

            if anonymous_identity is not None and anonymous_similarity is not None:
                status = "anonymous"
                if matched_gallery is not None:
                    if not read_only:
                        anonymous_identity.observation_count += 1
                    sample = None
                    _update_gallery(
                        matched_gallery,
                        embedding,
                        settings.reid_gallery_size,
                    )
                else:
                    sample = (
                        None
                        if read_only
                        else record_anonymous_observation(
                            session,
                            anonymous_identity,
                            embedding_list,
                            representative.detection_confidence,
                            anonymous_similarity,
                        )
                    )
                    in_video_galleries.append(
                        _InVideoIdentityGallery(
                            identity=anonymous_identity,
                            embeddings=[embedding.copy()],
                        )
                    )
            else:
                if read_only:
                    anonymous_identity = _ReadOnlyAnonymousIdentity(face_id=uuid4())
                    sample = None
                else:
                    anonymous_identity, sample = create_anonymous_identity(
                        session,
                        embedding_list,
                        representative.detection_confidence,
                        owner_user_id=owner_user_id,
                    )
                status = "new_anonymous"
                anonymous_similarity = None
                in_video_galleries.append(
                    _InVideoIdentityGallery(
                        identity=anonymous_identity,
                        embeddings=[embedding.copy()],
                    )
                )

            anonymous_changed = anonymous_changed or not read_only
            if sample is not None and not read_only:
                sample.image_path = save_anonymous_face_crop(
                    anonymous_identity.face_id,
                    representative.face_crop,
                )
                stored_anonymous_images.append(sample.image_path)
            if not read_only:
                synchronized_face_ids.add(anonymous_identity.face_id)
            results.append(
                RecognizedVideoTrack(
                    track_id=result_track_id,
                    face_id=anonymous_identity.face_id,
                    status=status,
                    recognized=False,
                    similarity=(
                        round(float(anonymous_similarity), 4)
                        if anonymous_similarity is not None
                        else None
                    ),
                    threshold=ANONYMOUS_MATCH_THRESHOLD,
                    person_id=None,
                    name=None,
                    metadata=None,
                    matched_image_path=(sample.image_path if sample is not None else None),
                    representative_frame_number=representative.frame_number,
                    representative_timestamp_ms=representative.timestamp_ms,
                    detection_confidence=round(
                        representative.detection_confidence, 4
                    ),
                    source_track_id=source_track_id,
                    first_seen_ms=first_seen_ms,
                    last_seen_ms=last_seen_ms,
                    observation_count=observation_count,
                )
            )

        if anonymous_changed and manage_transaction and not read_only:
            session.commit()
            for face_id in synchronized_face_ids:
                synchronize_face_id_safely(session, face_id)
    except Exception:
        session.rollback()
        for stored_path in stored_anonymous_images:
            delete_face_image(stored_path)
        raise

    return VideoRecognitionSummary(
        tracking=tracking,
        tracks=tuple(results),
        changed_face_ids=tuple(sorted(synchronized_face_ids, key=str)),
        stored_anonymous_image_paths=tuple(stored_anonymous_images),
        tracking_seconds=tracking_seconds,
        identity_seconds=time.perf_counter() - identity_started,
    )
