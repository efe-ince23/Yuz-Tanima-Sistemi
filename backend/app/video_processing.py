import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional, Tuple
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.face_storage import (
    delete_face_image,
    download_object_to_file,
    save_file_object,
)
from app.models import (
    RecognitionProcess,
    VideoAppearanceSegment,
    VideoFaceObservation,
    VideoJob,
    VideoTrack,
)
from app.vector_store import synchronize_face_id_safely
from app.video_config import get_video_settings
from app.video_config import get_video_tracking_settings
from app.video_recognition import (
    VideoRecognitionSummary,
    recognize_video_tracks,
)
from app.video_tracking import TrackedVideoFrame
from app.video_upload import normalize_live_recording


logger = logging.getLogger(__name__)

LIVE_OBSERVATION_MAX_GAP_MS = 2500
LIVE_OBSERVATION_GAP_MULTIPLIER = 2.4


class VideoProcessingError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class _Observation:
    track_number: int
    frame_number: int
    timestamp_ms: int
    bounding_box: Tuple[float, float, float, float]
    detection_confidence: float


def _normalize_live_job(
    session: Session,
    job: VideoJob,
    process: RecognitionProcess,
) -> None:
    if job.content_type != "video/webm":
        return

    old_object_path = job.object_path
    normalized_object_path = f"videos/{job.process_id}/source.mp4"
    stored_normalized = False
    with TemporaryDirectory(prefix="face-live-worker-") as directory:
        source_path = Path(directory) / "source.webm"
        download_object_to_file(old_object_path, source_path)
        normalized = normalize_live_recording(
            source_path,
            job.original_filename,
            get_video_settings(),
        )
        try:
            save_file_object(
                normalized_object_path,
                normalized.temporary_path,
                normalized.content_type,
            )
            stored_normalized = True
            metadata = normalized.metadata
            job.object_path = normalized_object_path
            job.content_type = normalized.content_type
            job.file_size_bytes = normalized.file_size_bytes
            job.duration_seconds = metadata.duration_seconds
            job.source_fps = metadata.source_fps
            job.width = metadata.width
            job.height = metadata.height
            job.frame_count = metadata.frame_count
            job.progress_percent = 2.0
            live_manifest = (process.task_detail or {}).get("live_manifest")
            process.task_detail = {
                "operation_type": "video_recognize",
                "source_type": "live_camera",
                "processed_face_count": 0,
                "faces": [],
                "status": "processing",
                "stage": "face_analysis",
                "live_manifest": live_manifest,
                "video": {
                    "original_filename": job.original_filename,
                    "object_path": normalized_object_path,
                    "duration_seconds": round(metadata.duration_seconds, 3),
                    "source_fps": round(metadata.source_fps, 3),
                    "width": metadata.width,
                    "height": metadata.height,
                    "frame_count": metadata.frame_count,
                    "container": metadata.container,
                    "codec": metadata.codec,
                },
            }
            session.commit()
        except Exception:
            session.rollback()
            if stored_normalized:
                try:
                    delete_face_image(normalized_object_path)
                except (OSError, ValueError):
                    logger.exception(
                        "Basarisiz canli video donusumu temizlenemedi: %s",
                        normalized_object_path,
                    )
            raise
        finally:
            normalized.cleanup()

    try:
        delete_face_image(old_object_path)
    except (OSError, ValueError):
        logger.exception("Ham canli kamera kaydi temizlenemedi: %s", old_object_path)


def _collect_observations(target: List[_Observation], frame: TrackedVideoFrame) -> None:
    for face in frame.faces:
        target.append(
            _Observation(
                track_number=face.track_id,
                frame_number=frame.frame_number,
                timestamp_ms=frame.timestamp_ms,
                bounding_box=face.detection.normalized_bounding_box,
                detection_confidence=face.detection.confidence,
            )
        )


def _track_result(track) -> Dict[str, object]:
    return {
        "trackId": track.track_id,
        "faceId": str(track.face_id),
        "status": track.status,
        "name": track.name,
        "metadata": track.metadata,
        "confidence": track.similarity,
        "firstSeenMs": None,
        "lastSeenMs": None,
    }


def _appearance_groups(
    observations: List[_Observation],
    max_gap_ms: int,
) -> List[List[_Observation]]:
    if not observations:
        return []
    ordered = sorted(observations, key=lambda item: (item.timestamp_ms, item.frame_number))
    groups = [[ordered[0]]]
    for observation in ordered[1:]:
        if observation.timestamp_ms - groups[-1][-1].timestamp_ms > max_gap_ms:
            groups.append([observation])
        else:
            groups[-1].append(observation)
    return groups


def _persist_result(
    session: Session,
    job: VideoJob,
    process: RecognitionProcess,
    summary: VideoRecognitionSummary,
    observations: List[_Observation],
) -> None:
    tracking_by_number = {
        track.track_id: track for track in summary.tracking.tracks
    }
    recognition_by_number = {track.track_id: track for track in summary.tracks}
    recognized_observations = []
    observations_by_track: Dict[int, List[_Observation]] = {}
    for item in observations:
        for recognized in summary.tracks:
            source_track_id = recognized.source_track_id or recognized.track_id
            tracked = tracking_by_number[source_track_id]
            first_seen_ms = (
                recognized.first_seen_ms
                if recognized.first_seen_ms is not None
                else tracked.first_seen_ms
            )
            last_seen_ms = (
                recognized.last_seen_ms
                if recognized.last_seen_ms is not None
                else tracked.last_seen_ms
            )
            if (
                item.track_number == source_track_id
                and first_seen_ms <= item.timestamp_ms <= last_seen_ms
            ):
                recognized_observations.append((item, recognized))
                observations_by_track.setdefault(recognized.track_id, []).append(item)
                break
    unique_faces = {track.face_id: track for track in summary.tracks}

    # A failed retry must never leave duplicate tracks or observations behind.
    session.execute(delete(VideoTrack).where(VideoTrack.process_id == job.process_id))
    persisted_tracks: Dict[int, VideoTrack] = {}
    result_tracks: List[Dict[str, object]] = []
    for recognized in summary.tracks:
        source_track_id = recognized.source_track_id or recognized.track_id
        tracked = tracking_by_number[source_track_id]
        track_observations = observations_by_track.get(recognized.track_id, [])
        first_seen_ms = (
            recognized.first_seen_ms
            if recognized.first_seen_ms is not None
            else tracked.first_seen_ms
        )
        last_seen_ms = (
            recognized.last_seen_ms
            if recognized.last_seen_ms is not None
            else tracked.last_seen_ms
        )
        video_track = VideoTrack(
            process_id=job.process_id,
            track_number=recognized.track_id,
            face_id=recognized.face_id,
            face_status=recognized.status,
            first_seen_ms=first_seen_ms,
            last_seen_ms=last_seen_ms,
            observation_count=len(track_observations),
            best_detection_confidence=(
                max(item.detection_confidence for item in track_observations)
                if track_observations
                else tracked.best_confidence
            ),
            best_recognition_confidence=recognized.similarity,
            best_frame_number=recognized.representative_frame_number,
            best_image_path=recognized.matched_image_path,
        )
        session.add(video_track)
        persisted_tracks[recognized.track_id] = video_track
        result_track = _track_result(recognized)
        result_track["firstSeenMs"] = first_seen_ms
        result_track["lastSeenMs"] = last_seen_ms
        result_tracks.append(result_track)

    session.flush()
    observation_rows = [
            VideoFaceObservation(
                process_id=job.process_id,
                track_id=persisted_tracks[recognized.track_id].id,
                face_id=recognized.face_id,
                face_status=recognized.status,
                frame_number=item.frame_number,
                timestamp_ms=item.timestamp_ms,
                bbox_x1=item.bounding_box[0],
                bbox_y1=item.bounding_box[1],
                bbox_x2=item.bounding_box[2],
                bbox_y2=item.bounding_box[3],
                detection_confidence=item.detection_confidence,
                recognition_confidence=recognized.similarity,
            )
            for item, recognized in recognized_observations
        ]
    segment_rows = []
    max_gap_ms = get_video_tracking_settings().max_gap_ms
    for track_number, track_observations in observations_by_track.items():
        recognized = recognition_by_number[track_number]
        video_track = persisted_tracks[track_number]
        for group in _appearance_groups(track_observations, max_gap_ms):
            segment_rows.append(
                VideoAppearanceSegment(
                    process_id=job.process_id,
                    track_id=video_track.id,
                    face_id=recognized.face_id,
                    face_status=recognized.status,
                    start_ms=group[0].timestamp_ms,
                    end_ms=group[-1].timestamp_ms,
                    start_frame=group[0].frame_number,
                    end_frame=group[-1].frame_number,
                    observation_count=len(group),
                    max_recognition_confidence=recognized.similarity,
                    average_recognition_confidence=recognized.similarity,
                )
            )
    session.add_all(observation_rows + segment_rows)

    detection = summary.tracking.detection
    now = datetime.now(timezone.utc)
    job.status = "completed"
    job.sampled_frame_count = detection.sampled_frame_count
    job.processed_frame_count = detection.sampled_frame_count
    job.progress_percent = 100.0
    job.detected_face_count = detection.detected_face_count
    job.unique_face_count = len(unique_faces)
    job.error_code = None
    job.error_detail = None
    job.completed_at = now

    result = {
        "processId": str(job.process_id),
        "status": "completed",
        "sampledFrameCount": detection.sampled_frame_count,
        "detectedFaceCount": detection.detected_face_count,
        "uniqueFaceCount": len(unique_faces),
        "tracks": result_tracks,
    }
    process.status = "completed"
    process.http_status = 200
    process.face_count = len(unique_faces)
    process.task_detail = {
        "operation_type": "video_recognize",
        "processed_face_count": len(unique_faces),
        "faces": [
            {
                "face_id": str(track.face_id),
                "status": track.status,
                "track_id": track.track_id,
            }
            for track in unique_faces.values()
        ],
        "status": "completed",
        "sampled_frame_count": detection.sampled_frame_count,
        "detected_face_count": detection.detected_face_count,
    }
    process.result = result
    process.error_detail = None
    process.completed_at = now


def _persist_live_manifest(
    session: Session,
    job: VideoJob,
    process: RecognitionProcess,
    manifest: Dict[str, object],
) -> bool:
    duration_ms = max(1, int(manifest.get("duration_ms") or 1))
    analysis_count = max(0, int(manifest.get("analysis_count") or 0))
    first_analysis_ms = manifest.get("first_analysis_ms")
    last_analysis_ms = manifest.get("last_analysis_ms")
    coverage_is_sufficient = (
        isinstance(first_analysis_ms, int)
        and isinstance(last_analysis_ms, int)
        and first_analysis_ms <= 2500
        and last_analysis_ms >= max(0, duration_ms - 2500)
        and analysis_count * 5000 >= duration_ms
    )
    if not coverage_is_sufficient:
        return False
    raw_observations = manifest.get("observations")
    if not isinstance(raw_observations, list) or not raw_observations:
        return False

    observations_by_face: Dict[UUID, List[Dict[str, object]]] = {}
    for raw in raw_observations:
        if not isinstance(raw, dict):
            continue
        try:
            face_id = UUID(str(raw["face_id"]))
        except (KeyError, TypeError, ValueError):
            continue
        observations_by_face.setdefault(face_id, []).append(raw)
    if not observations_by_face:
        return False

    session.execute(delete(VideoTrack).where(VideoTrack.process_id == job.process_id))
    session.flush()
    base_max_gap_ms = get_video_tracking_settings().max_gap_ms
    estimated_sample_interval_ms = max(
        1,
        round(duration_ms / max(1, analysis_count)),
    )
    max_gap_ms = min(
        LIVE_OBSERVATION_MAX_GAP_MS,
        max(
            base_max_gap_ms,
            round(
                estimated_sample_interval_ms
                * LIVE_OBSERVATION_GAP_MULTIPLIER
            ),
        ),
    )
    segment_padding_ms = min(
        max_gap_ms // 2,
        max(100, round(estimated_sample_interval_ms / 2)),
    )
    source_fps = job.source_fps or 25.0
    result_tracks: List[Dict[str, object]] = []
    total_observations = 0

    for track_number, (face_id, raw_items) in enumerate(
        sorted(observations_by_face.items(), key=lambda item: str(item[0])),
        start=1,
    ):
        items = sorted(raw_items, key=lambda item: int(item["timestamp_ms"]))
        statuses = {str(item.get("status")) for item in items}
        face_status = (
            "known"
            if "known" in statuses
            else "new_anonymous"
            if "new_anonymous" in statuses
            else "anonymous"
        )
        best = max(
            items,
            key=lambda item: (
                float(item.get("recognition_confidence") or -1.0),
                float(item.get("detection_confidence") or 0.0),
            ),
        )
        matched_image_url = best.get("matched_image_url")
        best_image_path = None
        if isinstance(matched_image_url, str) and matched_image_url.startswith("/media/"):
            best_image_path = matched_image_url[len("/media/"):]

        used_frames = set()
        prepared = []
        for item in items:
            timestamp_ms = max(0, int(item["timestamp_ms"]))
            frame_number = max(0, round(timestamp_ms * source_fps / 1000))
            while frame_number in used_frames:
                frame_number += 1
            used_frames.add(frame_number)
            prepared.append((item, timestamp_ms, frame_number))

        track = VideoTrack(
            process_id=job.process_id,
            track_number=track_number,
            face_id=face_id,
            face_status=face_status,
            first_seen_ms=prepared[0][1],
            last_seen_ms=prepared[-1][1],
            observation_count=len(prepared),
            best_detection_confidence=float(best.get("detection_confidence") or 0.0),
            best_recognition_confidence=(
                float(best["recognition_confidence"])
                if best.get("recognition_confidence") is not None
                else None
            ),
            best_frame_number=max(0, round(int(best["timestamp_ms"]) * source_fps / 1000)),
            best_image_path=best_image_path,
        )
        session.add(track)
        session.flush()

        observation_rows = []
        for item, timestamp_ms, frame_number in prepared:
            box = item["bounding_box"]
            observation_rows.append(
                VideoFaceObservation(
                    process_id=job.process_id,
                    track_id=track.id,
                    face_id=face_id,
                    face_status=str(item["status"]),
                    frame_number=frame_number,
                    timestamp_ms=timestamp_ms,
                    bbox_x1=float(box["x1"]),
                    bbox_y1=float(box["y1"]),
                    bbox_x2=float(box["x2"]),
                    bbox_y2=float(box["y2"]),
                    detection_confidence=float(item["detection_confidence"]),
                    recognition_confidence=(
                        float(item["recognition_confidence"])
                        if item.get("recognition_confidence") is not None
                        else None
                    ),
                )
            )
        session.add_all(observation_rows)

        groups = [[prepared[0]]]
        for prepared_item in prepared[1:]:
            if prepared_item[1] - groups[-1][-1][1] > max_gap_ms:
                groups.append([prepared_item])
            else:
                groups[-1].append(prepared_item)
        for group in groups:
            segment_start_ms = max(0, group[0][1] - segment_padding_ms)
            segment_end_ms = min(
                duration_ms,
                group[-1][1] + segment_padding_ms,
            )
            recognition_scores = [
                float(item[0]["recognition_confidence"])
                for item in group
                if item[0].get("recognition_confidence") is not None
            ]
            session.add(
                VideoAppearanceSegment(
                    process_id=job.process_id,
                    track_id=track.id,
                    face_id=face_id,
                    face_status=face_status,
                    start_ms=segment_start_ms,
                    end_ms=segment_end_ms,
                    start_frame=max(
                        0,
                        round(segment_start_ms * source_fps / 1000),
                    ),
                    end_frame=max(
                        0,
                        round(segment_end_ms * source_fps / 1000),
                    ),
                    observation_count=len(group),
                    max_recognition_confidence=(
                        max(recognition_scores) if recognition_scores else None
                    ),
                    average_recognition_confidence=(
                        sum(recognition_scores) / len(recognition_scores)
                        if recognition_scores
                        else None
                    ),
                )
            )

        total_observations += len(prepared)
        result_tracks.append(
            {
                "trackId": track_number,
                "faceId": str(face_id),
                "status": face_status,
                "name": best.get("name") if face_status == "known" else None,
                "metadata": best.get("metadata") if face_status == "known" else None,
                "confidence": track.best_recognition_confidence,
                "firstSeenMs": track.first_seen_ms,
                "lastSeenMs": track.last_seen_ms,
            }
        )

    now = datetime.now(timezone.utc)
    job.status = "completed"
    job.sampled_frame_count = analysis_count
    job.processed_frame_count = analysis_count
    job.progress_percent = 100.0
    job.detected_face_count = total_observations
    job.unique_face_count = len(observations_by_face)
    job.error_code = None
    job.error_detail = None
    job.completed_at = now
    process.status = "completed"
    process.http_status = 200
    process.face_count = len(observations_by_face)
    process.task_detail = {
        "operation_type": "video_recognize",
        "source_type": "live_camera",
        "processed_face_count": len(observations_by_face),
        "faces": [
            {
                "face_id": track["faceId"],
                "status": track["status"],
                "track_id": track["trackId"],
            }
            for track in result_tracks
        ],
        "status": "completed",
        "stage": "live_manifest_completed",
        "sampled_frame_count": analysis_count,
        "detected_face_count": total_observations,
    }
    process.result = {
        "processId": str(job.process_id),
        "status": "completed",
        "sampledFrameCount": analysis_count,
        "detectedFaceCount": total_observations,
        "uniqueFaceCount": len(observations_by_face),
        "tracks": result_tracks,
    }
    process.error_detail = None
    process.completed_at = now
    return True


def _mark_failed(session: Session, process_id: UUID, error: Exception) -> None:
    try:
        job = session.get(VideoJob, process_id)
        process = session.get(RecognitionProcess, process_id)
        now = datetime.now(timezone.utc)
        detail = str(error)[:4000]
        if job is not None:
            job.status = "failed"
            job.error_code = "VIDEO_PROCESSING_FAILED"
            job.error_detail = detail
            job.completed_at = now
        if process is not None:
            process.status = "failed"
            process.http_status = 500
            process.error_detail = detail
            process.completed_at = now
            process.task_detail = {
                "operation_type": "video_recognize",
                "processed_face_count": 0,
                "faces": [],
                "status": "failed",
            }
        session.commit()
    except Exception:
        session.rollback()
        logger.exception("Video hata durumu kaydedilemedi: %s", process_id)


def process_video_job(
    session: Session,
    process_id: UUID,
    sample_fps: Optional[float] = None,
) -> VideoJob:
    job = session.scalar(
        select(VideoJob)
        .where(VideoJob.process_id == process_id)
        .with_for_update()
    )
    if job is None:
        raise VideoProcessingError("VIDEO_JOB_NOT_FOUND", "Video isi bulunamadi.")
    if job.status == "completed":
        return job
    if job.status == "processing":
        raise VideoProcessingError(
            "VIDEO_JOB_ALREADY_PROCESSING", "Video isi zaten isleniyor."
        )
    if job.status == "cancelled":
        raise VideoProcessingError("VIDEO_JOB_CANCELLED", "Video isi iptal edildi.")

    process = job.process
    job.status = "processing"
    job.progress_percent = 0.0
    job.error_code = None
    job.error_detail = None
    job.started_at = datetime.now(timezone.utc)
    job.completed_at = None
    process.status = "processing"
    process.http_status = None
    process.error_detail = None
    process.completed_at = None
    session.commit()

    observations: List[_Observation] = []
    stored_paths: Tuple[str, ...] = ()
    try:
        live_manifest = (process.task_detail or {}).get("live_manifest")
        _normalize_live_job(session, job, process)
        if isinstance(live_manifest, dict) and _persist_live_manifest(
            session,
            job,
            process,
            live_manifest,
        ):
            session.commit()
            session.refresh(job)
            return job
        summary = recognize_video_tracks(
            session,
            job.object_path,
            sample_fps or get_video_settings().sample_fps,
            frame_handler=lambda frame: _collect_observations(observations, frame),
            manage_transaction=False,
            owner_user_id=process.owner_user_id,
        )
        stored_paths = summary.stored_anonymous_image_paths
        _persist_result(session, job, process, summary, observations)
        session.commit()
        for face_id in summary.changed_face_ids:
            synchronize_face_id_safely(session, face_id)
        session.refresh(job)
        return job
    except Exception as error:
        session.rollback()
        for stored_path in stored_paths:
            try:
                delete_face_image(stored_path)
            except (OSError, ValueError):
                logger.exception("Anonim video yuzu temizlenemedi: %s", stored_path)
        _mark_failed(session, process_id, error)
        if isinstance(error, VideoProcessingError):
            raise
        raise VideoProcessingError(
            "VIDEO_PROCESSING_FAILED", "Video islenemedi."
        ) from error
