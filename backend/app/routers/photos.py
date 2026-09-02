from uuid import UUID

from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api_errors import api_error
from app.auth import get_current_user
from app.database import get_database_session
from app.face_storage import read_face_image
from app.models import RecognitionProcess, User
from app.schemas import FaceIdentifyResponse, PhotoHistoryItemResponse, PhotoHistoryListResponse


router = APIRouter(prefix="/api/photos", tags=["photos"])


def _visible_photo(session: Session, process_id: UUID, user: User) -> RecognitionProcess:
    statement = select(RecognitionProcess).where(
        RecognitionProcess.process_id == process_id,
        RecognitionProcess.operation_type == "identify",
        RecognitionProcess.source_image_path.is_not(None),
    )
    if user.role != "admin":
        statement = statement.where(RecognitionProcess.owner_user_id == user.id)
    process = session.scalar(statement)
    if process is None:
        raise api_error(404, "PHOTO_HISTORY_NOT_FOUND", "Fotograf gecmisi bulunamadi.")
    return process


def _history_item(process: RecognitionProcess, owner: User) -> PhotoHistoryItemResponse:
    result = None
    if process.result:
        try:
            result = FaceIdentifyResponse.model_validate(process.result)
        except ValidationError:
            result = None
    return PhotoHistoryItemResponse(
        process_id=process.process_id,
        status=process.status,
        face_count=process.face_count,
        original_filename=process.source_filename,
        owner_username=owner.username if owner is not None else None,
        owner_full_name=owner.full_name if owner is not None else None,
        image_url=f"/api/photos/{process.process_id}/content",
        image_width=process.source_image_width,
        image_height=process.source_image_height,
        created_at=process.created_at,
        completed_at=process.completed_at,
        result=result,
    )


@router.get("", response_model=PhotoHistoryListResponse, include_in_schema=False)
def list_photo_history(
    limit: int = Query(12, ge=1, le=50),
    offset: int = Query(0, ge=0),
    user: User = Depends(get_current_user),
    session: Session = Depends(get_database_session),
) -> PhotoHistoryListResponse:
    filters = [
        RecognitionProcess.owner_user_id == user.id,
        RecognitionProcess.operation_type == "identify",
        RecognitionProcess.source_image_path.is_not(None),
    ]
    total = session.scalar(select(func.count()).select_from(RecognitionProcess).where(*filters)) or 0
    rows = session.execute(
        select(RecognitionProcess, User)
        .outerjoin(User, User.id == RecognitionProcess.owner_user_id)
        .where(*filters)
        .order_by(RecognitionProcess.created_at.desc(), RecognitionProcess.process_id.desc())
        .offset(offset)
        .limit(limit)
    ).all()
    return PhotoHistoryListResponse(
        total=total,
        limit=limit,
        offset=offset,
        items=[_history_item(process, owner) for process, owner in rows],
    )


@router.get("/{process_id}/content", include_in_schema=False)
def get_photo_content(
    process_id: UUID,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_database_session),
) -> Response:
    process = _visible_photo(session, process_id, user)
    try:
        content, content_type = read_face_image(process.source_image_path)
    except FileNotFoundError as error:
        raise api_error(404, "PHOTO_CONTENT_NOT_FOUND", "Fotograf dosyasi bulunamadi.") from error
    except OSError as error:
        raise api_error(503, "PHOTO_STORAGE_UNAVAILABLE", "Fotograf deposuna erisilemedi.") from error
    return Response(
        content=content,
        media_type=content_type,
        headers={"Cache-Control": "private, max-age=300"},
    )
