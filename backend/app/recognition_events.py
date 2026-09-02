import logging
from typing import Optional
from uuid import UUID

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.models import RecognitionEvent


logger = logging.getLogger(__name__)


def record_recognition_event(
    session: Session,
    *,
    recognized: bool,
    process_id: Optional[UUID],
    person_id: Optional[int],
    face_id: Optional[UUID],
    face_status: Optional[str],
    similarity: Optional[float],
    threshold: float,
    commit: bool = True,
) -> None:
    event = RecognitionEvent(
        process_id=process_id,
        recognized=recognized,
        person_id=person_id if recognized else None,
        face_id=face_id,
        face_status=face_status,
        similarity=similarity,
        threshold=threshold,
    )
    try:
        session.add(event)
        if commit:
            session.commit()
    except SQLAlchemyError:
        session.rollback()
        logger.exception("Recognition event could not be recorded")
