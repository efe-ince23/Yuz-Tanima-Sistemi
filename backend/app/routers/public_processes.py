from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_database_session
from app.auth import get_current_user
from app.models import User
from app.routers.processes import get_process
from app.schemas import RecognitionProcessResponse


router = APIRouter(prefix="/processes", tags=["processes"])


@router.get(
    "/{processId}",
    response_model=RecognitionProcessResponse,
    summary="Get process details",
)
def get_public_process(
    processId: UUID,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_database_session),
) -> RecognitionProcessResponse:
    return get_process(processId, user, session)
