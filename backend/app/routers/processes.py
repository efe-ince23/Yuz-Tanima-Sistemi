from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.api_errors import api_error
from app.auth import get_current_user
from app.database import get_database_session
from app.models import RecognitionProcess, User
from app.process_tracking import read_fallback_process
from app.schemas import RecognitionProcessEventResponse, RecognitionProcessResponse


router = APIRouter(prefix="/api/processes", tags=["processes"])


@router.get("/{process_id}", response_model=RecognitionProcessResponse)
def get_process(
    process_id: UUID,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_database_session),
) -> RecognitionProcessResponse:
    fallback = read_fallback_process(process_id)
    process = session.scalar(
        select(RecognitionProcess)
        .options(selectinload(RecognitionProcess.events))
        .where(RecognitionProcess.process_id == process_id)
    )
    if process is None and fallback is None:
        raise api_error(404, "PROCESS_NOT_FOUND", "Islem kaydi bulunamadi.")
    if process is not None and user.role != "admin" and process.owner_user_id != user.id:
        raise api_error(404, "PROCESS_NOT_FOUND", "Islem kaydi bulunamadi.")
    if process is None and user.role != "admin":
        raise api_error(404, "PROCESS_NOT_FOUND", "Islem kaydi bulunamadi.")

    events = sorted(process.events, key=lambda event: event.id) if process else []
    use_fallback = fallback is not None and fallback.get("status") != "processing"
    if use_fallback:
        operation_type = fallback["operation_type"]
        process_status = fallback["status"]
        http_status = fallback.get("http_status")
        face_count = fallback.get("face_count", 0)
        task_detail = fallback.get("task_detail")
        result = fallback.get("result")
        error_detail = fallback.get("error_detail")
        created_at = fallback["created_at"]
        completed_at = fallback.get("completed_at")
    elif process is not None:
        operation_type = process.operation_type
        process_status = process.status
        http_status = process.http_status
        face_count = process.face_count
        task_detail = process.task_detail
        result = process.result
        error_detail = process.error_detail
        created_at = process.created_at
        completed_at = process.completed_at
    else:
        operation_type = fallback["operation_type"]
        process_status = fallback["status"]
        http_status = fallback.get("http_status")
        face_count = fallback.get("face_count", 0)
        task_detail = fallback.get("task_detail")
        result = fallback.get("result")
        error_detail = fallback.get("error_detail")
        created_at = fallback["created_at"]
        completed_at = fallback.get("completed_at")

    return RecognitionProcessResponse(
        process_id=process_id,
        operation_type=operation_type,
        status=process_status,
        http_status=http_status,
        face_count=face_count,
        task_detail=task_detail,
        result=result,
        error_detail=error_detail,
        created_at=created_at,
        completed_at=completed_at,
        events=[
            RecognitionProcessEventResponse(
                id=event.id,
                face_id=event.face_id,
                face_status=event.face_status,
                recognized=event.recognized,
                person_id=event.person_id,
                similarity=event.similarity,
                threshold=event.threshold,
                created_at=event.created_at,
            )
            for event in events
        ],
    )
