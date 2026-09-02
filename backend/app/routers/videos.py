import logging
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, Query, Request, UploadFile, status
from fastapi.responses import Response, StreamingResponse
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.api_errors import api_error
from app.auth import get_current_user
from app.database import get_database_session
from app.face_storage import (
    delete_face_image,
    finalize_staged_files,
    image_url,
    object_size,
    restore_staged_files,
    save_file_object,
    stage_face_images_for_deletion,
    stream_object,
)
from app.models import Person, RecognitionProcess, User, VideoJob, VideoTrack
from app.schemas import (
    LiveVideoManifestInput,
    VideoAppearanceSegmentResponse,
    VideoBoundingBoxResponse,
    VideoFaceHistoryItemResponse,
    VideoFaceHistoryResponse,
    VideoJobResponse,
    VideoJobListResponse,
    VideoObservationResponse,
    VideoResultResponse,
    VideoTrackResultResponse,
)
from app.video_config import get_video_settings
from app.video_upload import (
    ReceivedLiveRecording,
    ValidatedVideo,
    VideoUploadError,
    receive_live_recording,
    validate_uploaded_video,
)
from app.video_streaming import InvalidByteRange, parse_byte_range
from app.video_worker import submit_video_job


router = APIRouter(prefix="/api/videos", tags=["videos"])
logger = logging.getLogger(__name__)


def _video_job_response(job: VideoJob) -> VideoJobResponse:
    return VideoJobResponse.model_validate(job)


def _video_track_result(
    track: VideoTrack,
    person: Optional[Person] = None,
) -> VideoTrackResultResponse:
    appearance_segments = sorted(
        track.appearance_segments,
        key=lambda item: (item.start_ms, item.id),
    )
    observations = sorted(
        track.observations,
        key=lambda item: (item.timestamp_ms, item.frame_number),
    )
    return VideoTrackResultResponse(
        track_id=track.track_number,
        face_id=track.face_id,
        status=track.face_status,
        name=(
            f"{person.first_name} {person.last_name}"
            if person is not None
            else None
        ),
        metadata=(
            {"description": person.description}
            if person is not None
            else None
        ),
        first_seen_ms=track.first_seen_ms,
        last_seen_ms=track.last_seen_ms,
        observation_count=track.observation_count,
        best_detection_confidence=track.best_detection_confidence,
        best_recognition_confidence=track.best_recognition_confidence,
        best_frame_number=track.best_frame_number,
        best_image_url=(
            image_url(track.best_image_path)
            if track.best_image_path
            else None
        ),
        appearances=[
            VideoAppearanceSegmentResponse(
                start_ms=segment.start_ms,
                end_ms=segment.end_ms,
                start_frame=segment.start_frame,
                end_frame=segment.end_frame,
                observation_count=segment.observation_count,
                max_recognition_confidence=segment.max_recognition_confidence,
                average_recognition_confidence=(
                    segment.average_recognition_confidence
                ),
            )
            for segment in appearance_segments
        ],
        observations=[
            VideoObservationResponse(
                frame_number=observation.frame_number,
                timestamp_ms=observation.timestamp_ms,
                bounding_box=VideoBoundingBoxResponse(
                    x1=observation.bbox_x1,
                    y1=observation.bbox_y1,
                    x2=observation.bbox_x2,
                    y2=observation.bbox_y2,
                ),
                detection_confidence=observation.detection_confidence,
                recognition_confidence=observation.recognition_confidence,
            )
            for observation in observations
        ],
    )


def _visible_video_job(
    session: Session,
    process_id: UUID,
    user: object,
) -> VideoJob:
    # Direct service-level callers from the existing test suite do not resolve
    # FastAPI dependencies; they retain the pre-authentication admin behavior.
    if not isinstance(user, User):
        job = session.get(VideoJob, process_id)
        if job is None:
            raise api_error(404, "VIDEO_JOB_NOT_FOUND", "Video is kaydi bulunamadi.")
        return job
    statement = (
        select(VideoJob)
        .join(RecognitionProcess, RecognitionProcess.process_id == VideoJob.process_id)
        .where(VideoJob.process_id == process_id)
    )
    if user.role != "admin":
        statement = statement.where(RecognitionProcess.owner_user_id == user.id)
    job = session.scalar(statement)
    if job is None:
        raise api_error(404, "VIDEO_JOB_NOT_FOUND", "Video is kaydi bulunamadi.")
    return job


def _delete_stored_video_safely(object_path: str) -> None:
    try:
        delete_face_image(object_path)
    except (OSError, ValueError):
        logger.exception("Basarisiz video yuklemesi temizlenemedi: %s", object_path)


def _submit_video_safely(process_id: UUID) -> None:
    try:
        submit_video_job(process_id)
    except RuntimeError:
        # The durable queued row remains recoverable after an application restart.
        logger.exception("Video isi arka plan islemcisine iletilemedi: %s", process_id)


def _persist_video_job(
    process_id: UUID,
    validated: ValidatedVideo,
    session: Session,
    source_type: str,
) -> VideoJobResponse:
    object_path = f"videos/{process_id}/source.mp4"
    stored_path = None
    try:
        stored_path = save_file_object(
            object_path,
            validated.temporary_path,
            validated.content_type,
        )
        process = session.get(RecognitionProcess, process_id)
        if process is None:
            raise RuntimeError("Video process kaydi olusturulamadi.")

        metadata = validated.metadata
        job = VideoJob(
            process_id=process_id,
            status="queued",
            original_filename=validated.original_filename,
            object_path=stored_path,
            content_type=validated.content_type,
            file_size_bytes=validated.file_size_bytes,
            duration_seconds=metadata.duration_seconds,
            source_fps=metadata.source_fps,
            width=metadata.width,
            height=metadata.height,
            frame_count=metadata.frame_count,
        )
        session.add(job)
        process.status = "queued"
        process.http_status = status.HTTP_202_ACCEPTED
        process.task_detail = {
            "operation_type": "video_recognize",
            "source_type": source_type,
            "processed_face_count": 0,
            "faces": [],
            "status": "queued",
            "video": {
                "original_filename": validated.original_filename,
                "object_path": stored_path,
                "duration_seconds": round(metadata.duration_seconds, 3),
                "source_fps": round(metadata.source_fps, 3),
                "width": metadata.width,
                "height": metadata.height,
                "frame_count": metadata.frame_count,
                "container": metadata.container,
                "codec": metadata.codec,
            },
        }
        session.flush()
        response = _video_job_response(job)
        process.result = response.model_dump(mode="json")
        session.commit()
        _submit_video_safely(process_id)
        return response
    except (OSError, RuntimeError, ValueError) as error:
        session.rollback()
        if stored_path is not None:
            _delete_stored_video_safely(stored_path)
        raise api_error(
            503,
            "VIDEO_STORAGE_UNAVAILABLE",
            "Video kalici olarak saklanamadi.",
        ) from error
    except SQLAlchemyError as error:
        session.rollback()
        if stored_path is not None:
            _delete_stored_video_safely(stored_path)
        raise api_error(
            500,
            "VIDEO_JOB_CREATE_FAILED",
            "Video is kaydi olusturulamadi.",
        ) from error
    finally:
        validated.cleanup()


def _persist_live_recording_job(
    process_id: UUID,
    received: ReceivedLiveRecording,
    manifest: LiveVideoManifestInput,
    session: Session,
) -> VideoJobResponse:
    object_path = f"videos/{process_id}/source.webm"
    stored_path = None
    try:
        stored_path = save_file_object(
            object_path,
            received.temporary_path,
            received.content_type,
        )
        process = session.get(RecognitionProcess, process_id)
        if process is None:
            raise RuntimeError("Video process kaydi olusturulamadi.")

        job = VideoJob(
            process_id=process_id,
            status="queued",
            original_filename=f"{received.original_filename.rsplit('.', 1)[0]}.mp4",
            object_path=stored_path,
            content_type=received.content_type,
            file_size_bytes=received.file_size_bytes,
            duration_seconds=None,
            source_fps=None,
            width=None,
            height=None,
            frame_count=None,
        )
        session.add(job)
        process.status = "queued"
        process.http_status = status.HTTP_202_ACCEPTED
        process.task_detail = {
            "operation_type": "video_recognize",
            "source_type": "live_camera",
            "processed_face_count": 0,
            "faces": [],
            "status": "queued",
            "stage": "awaiting_normalization",
            "live_manifest": manifest.model_dump(mode="json"),
            "video": {
                "original_filename": job.original_filename,
                "object_path": stored_path,
                "container": "webm",
            },
        }
        session.flush()
        response = _video_job_response(job)
        process.result = response.model_dump(mode="json")
        session.commit()
        _submit_video_safely(process_id)
        return response
    except (OSError, RuntimeError, ValueError) as error:
        session.rollback()
        if stored_path is not None:
            _delete_stored_video_safely(stored_path)
        raise api_error(
            503,
            "VIDEO_STORAGE_UNAVAILABLE",
            "Canli kamera kaydi kalici olarak saklanamadi.",
        ) from error
    except SQLAlchemyError as error:
        session.rollback()
        if stored_path is not None:
            _delete_stored_video_safely(stored_path)
        raise api_error(
            500,
            "VIDEO_JOB_CREATE_FAILED",
            "Canli kamera video isi olusturulamadi.",
        ) from error
    finally:
        received.cleanup()


@router.get(
    "",
    response_model=VideoJobListResponse,
    include_in_schema=False,
)
def list_video_jobs(
    limit: int = Query(20, ge=1, le=50),
    offset: int = Query(0, ge=0),
    user: User = Depends(get_current_user),
    session: Session = Depends(get_database_session),
) -> VideoJobListResponse:
    base_filter = [] if user.role == "admin" else [RecognitionProcess.owner_user_id == user.id]
    total = session.scalar(
        select(func.count())
        .select_from(VideoJob)
        .join(RecognitionProcess, RecognitionProcess.process_id == VideoJob.process_id)
        .where(*base_filter)
    ) or 0
    jobs = list(
        session.scalars(
            select(VideoJob)
            .join(RecognitionProcess, RecognitionProcess.process_id == VideoJob.process_id)
            .where(*base_filter)
            .order_by(VideoJob.created_at.desc(), VideoJob.process_id.desc())
            .offset(offset)
            .limit(limit)
        ).all()
    )
    return VideoJobListResponse(
        total=total,
        limit=limit,
        offset=offset,
        items=[_video_job_response(job) for job in jobs],
    )


@router.get(
    "/faces/{face_id}/history",
    response_model=VideoFaceHistoryResponse,
    include_in_schema=False,
)
def get_face_video_history(
    face_id: UUID,
    limit: int = Query(20, ge=1, le=50),
    offset: int = Query(0, ge=0),
    user: User = Depends(get_current_user),
    session: Session = Depends(get_database_session),
) -> VideoFaceHistoryResponse:
    owner_filter = [] if user.role == "admin" else [RecognitionProcess.owner_user_id == user.id]
    total = session.scalar(
        select(func.count(func.distinct(VideoTrack.process_id)))
        .join(RecognitionProcess, RecognitionProcess.process_id == VideoTrack.process_id)
        .where(VideoTrack.face_id == face_id, *owner_filter)
    ) or 0
    jobs = list(
        session.scalars(
            select(VideoJob)
            .join(VideoTrack, VideoTrack.process_id == VideoJob.process_id)
            .join(RecognitionProcess, RecognitionProcess.process_id == VideoJob.process_id)
            .where(VideoTrack.face_id == face_id, *owner_filter)
            .distinct()
            .order_by(VideoJob.created_at.desc())
            .offset(offset)
            .limit(limit)
        ).all()
    )
    items = []
    for job in jobs:
        tracks = list(
            session.scalars(
                select(VideoTrack)
                .options(selectinload(VideoTrack.appearance_segments))
                .where(
                    VideoTrack.process_id == job.process_id,
                    VideoTrack.face_id == face_id,
                )
                .order_by(VideoTrack.track_number)
            ).all()
        )
        appearances = [
            VideoAppearanceSegmentResponse(
                start_ms=segment.start_ms,
                end_ms=segment.end_ms,
                start_frame=segment.start_frame,
                end_frame=segment.end_frame,
                observation_count=segment.observation_count,
                max_recognition_confidence=segment.max_recognition_confidence,
                average_recognition_confidence=segment.average_recognition_confidence,
            )
            for track in tracks
            for segment in sorted(
                track.appearance_segments,
                key=lambda item: (item.start_ms, item.id),
            )
        ]
        appearances.sort(key=lambda item: (item.start_ms, item.end_ms))
        items.append(
            VideoFaceHistoryItemResponse(
                process_id=job.process_id,
                original_filename=job.original_filename,
                created_at=job.created_at,
                duration_seconds=job.duration_seconds,
                first_seen_ms=min(track.first_seen_ms for track in tracks),
                last_seen_ms=max(track.last_seen_ms for track in tracks),
                observation_count=sum(track.observation_count for track in tracks),
                appearances=appearances,
            )
        )
    return VideoFaceHistoryResponse(
        face_id=face_id,
        total=total,
        limit=limit,
        offset=offset,
        items=items,
    )


@router.post(
    "",
    response_model=VideoJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    include_in_schema=False,
)
async def upload_video(
    request: Request,
    file: UploadFile = File(
        ...,
        description="H.264 codec kullanan MP4 video.",
    ),
    user: User = Depends(get_current_user),
    session: Session = Depends(get_database_session),
) -> VideoJobResponse:
    process_id: UUID = request.state.process_id
    try:
        validated = await validate_uploaded_video(file, get_video_settings())
    except VideoUploadError as error:
        raise api_error(
            error.status_code,
            error.code,
            error.message,
            error.details,
        ) from error
    finally:
        await file.close()

    return _persist_video_job(process_id, validated, session, "uploaded_file")


@router.post(
    "/live-recordings",
    response_model=VideoJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    include_in_schema=False,
)
async def upload_live_video_recording(
    request: Request,
    file: UploadFile = File(
        ...,
        description="Tarayicida MediaRecorder ile olusturulan kamera kaydi.",
    ),
    manifest: str = Form(..., description="Canli analiz zaman ve yuz sonuclari."),
    user: User = Depends(get_current_user),
    session: Session = Depends(get_database_session),
) -> VideoJobResponse:
    process_id: UUID = request.state.process_id
    if len(manifest.encode("utf-8")) > 2 * 1024 * 1024:
        raise api_error(
            413,
            "LIVE_MANIFEST_TOO_LARGE",
            "Canli analiz bilgisi izin verilen boyutu asiyor.",
        )
    try:
        parsed_manifest = LiveVideoManifestInput.model_validate_json(manifest)
    except ValidationError as error:
        raise api_error(
            422,
            "LIVE_MANIFEST_INVALID",
            "Canli analiz bilgisi gecersiz.",
            {"errors": error.errors(include_url=False, include_context=False)},
        ) from error
    if parsed_manifest.duration_ms > get_video_settings().max_duration_seconds * 1000:
        raise api_error(
            413,
            "VIDEO_DURATION_EXCEEDED",
            "Canli kamera kaydi izin verilen sureden uzun.",
        )
    try:
        received = await receive_live_recording(file, get_video_settings())
    except VideoUploadError as error:
        raise api_error(
            error.status_code,
            error.code,
            error.message,
            error.details,
        ) from error
    finally:
        await file.close()

    return _persist_live_recording_job(process_id, received, parsed_manifest, session)


@router.get(
    "/{process_id}",
    response_model=VideoJobResponse,
    include_in_schema=False,
)
def get_video_job(
    process_id: UUID,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_database_session),
) -> VideoJobResponse:
    job = _visible_video_job(session, process_id, user)
    return _video_job_response(job)


@router.delete("/{process_id}", status_code=status.HTTP_204_NO_CONTENT, include_in_schema=False)
def delete_video_job(
    process_id: UUID,
    session: Session = Depends(get_database_session),
    user: User = Depends(get_current_user),
) -> Response:
    job = _visible_video_job(session, process_id, user)
    if job.status in {"queued", "processing"}:
        raise api_error(
            409,
            "VIDEO_JOB_ACTIVE",
            "Devam eden video islemi silinemez.",
            {"status": job.status},
        )

    try:
        staged_files = stage_face_images_for_deletion([job.object_path])
    except (OSError, ValueError) as error:
        raise api_error(
            503,
            "VIDEO_FILE_STAGE_FAILED",
            "Video dosyasi silmeye hazirlanamadi.",
        ) from error

    try:
        process = session.get(RecognitionProcess, process_id)
        if process is None:
            session.delete(job)
        else:
            session.delete(process)
        session.commit()
    except SQLAlchemyError as error:
        session.rollback()
        try:
            restore_staged_files(staged_files)
        except OSError:
            logger.exception("Video dosyasi geri getirilemedi: %s", process_id)
        raise api_error(
            500,
            "VIDEO_JOB_DELETE_FAILED",
            "Video is kaydi silinemedi.",
        ) from error

    try:
        finalize_staged_files(staged_files)
    except OSError:
        logger.exception("Video gecici dosyalari temizlenemedi: %s", process_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/{process_id}/content", include_in_schema=False)
def stream_video_content(
    process_id: UUID,
    request: Request,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_database_session),
):
    job = _visible_video_job(session, process_id, user)
    try:
        size = object_size(job.object_path)
    except FileNotFoundError as error:
        raise api_error(404, "VIDEO_CONTENT_NOT_FOUND", "Video dosyasi bulunamadi.") from error
    except OSError as error:
        raise api_error(503, "VIDEO_STORAGE_UNAVAILABLE", "Video deposuna erisilemedi.") from error

    try:
        requested_range = parse_byte_range(request.headers.get("range"), size)
    except InvalidByteRange:
        return Response(
            status_code=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
            headers={"Content-Range": f"bytes */{size}", "Accept-Ranges": "bytes"},
        )

    if requested_range is None:
        start, end, response_status = 0, size - 1, status.HTTP_200_OK
    else:
        start, end = requested_range.start, requested_range.end
        response_status = status.HTTP_206_PARTIAL_CONTENT
    length = end - start + 1
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(length),
    }
    if requested_range is not None:
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    return StreamingResponse(
        stream_object(job.object_path, offset=start, length=length),
        status_code=response_status,
        media_type=job.content_type,
        headers=headers,
    )


@router.get(
    "/{process_id}/result",
    response_model=VideoResultResponse,
    include_in_schema=False,
)
def get_video_result(
    process_id: UUID,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_database_session),
) -> VideoResultResponse:
    job = _visible_video_job(session, process_id, user)
    if job.status != "completed":
        code = (
            "VIDEO_PROCESSING_FAILED"
            if job.status == "failed"
            else "VIDEO_RESULT_NOT_READY"
        )
        message = (
            "Video islenemedi."
            if job.status == "failed"
            else "Video sonucu henuz hazir degil."
        )
        raise api_error(
            409,
            code,
            message,
            {
                "status": job.status,
                "progress_percent": job.progress_percent,
                "error_detail": job.error_detail,
            },
        )

    tracks = list(
        session.scalars(
            select(VideoTrack)
            .options(
                selectinload(VideoTrack.observations),
                selectinload(VideoTrack.appearance_segments),
            )
            .where(VideoTrack.process_id == process_id)
            .order_by(VideoTrack.track_number)
        ).all()
    )
    known_face_ids = [
        track.face_id
        for track in tracks
        if track.face_status == "known" and track.face_id is not None
    ]
    people_by_face_id = {
        person.face_id: person
        for person in session.scalars(
            select(Person).where(Person.face_id.in_(known_face_ids))
        ).all()
    }
    result_tracks = [
        _video_track_result(track, people_by_face_id.get(track.face_id))
        for track in tracks
    ]

    return VideoResultResponse(
        process_id=job.process_id,
        status="completed",
        video_url=f"/api/videos/{job.process_id}/content",
        duration_seconds=job.duration_seconds,
        sampled_frame_count=job.sampled_frame_count,
        detected_face_count=job.detected_face_count,
        unique_face_count=job.unique_face_count,
        tracks=result_tracks,
    )
