from fastapi import APIRouter, Depends
from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from app.database import get_database_session
from app.auth import get_current_user
from app.models import RecognitionEvent, RecognitionProcess, User
from app.schemas import RecognitionStatisticsResponse


router = APIRouter(prefix="/api/statistics", tags=["statistics"])


@router.get("", response_model=RecognitionStatisticsResponse)
def get_recognition_statistics(
    user: User = Depends(get_current_user),
    session: Session = Depends(get_database_session),
) -> RecognitionStatisticsResponse:
    query = select(
        func.count(RecognitionEvent.id),
        func.coalesce(
            func.sum(case((RecognitionEvent.recognized.is_(True), 1), else_=0)),
            0,
        ),
        func.max(RecognitionEvent.created_at),
    )
    if user.role != "admin":
        query = query.join(
            RecognitionProcess,
            RecognitionProcess.process_id == RecognitionEvent.process_id,
        ).where(RecognitionProcess.owner_user_id == user.id)
    total_operations, recognized_count, latest_event_at = session.execute(query).one()
    total = int(total_operations)
    recognized = int(recognized_count)
    unrecognized = total - recognized
    success_rate = round((recognized / total * 100) if total else 0.0, 2)

    return RecognitionStatisticsResponse(
        total_operations=total,
        recognized_count=recognized,
        unrecognized_count=unrecognized,
        success_rate=success_rate,
        latest_event_at=latest_event_at,
    )
