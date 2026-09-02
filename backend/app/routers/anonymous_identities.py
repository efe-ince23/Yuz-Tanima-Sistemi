from uuid import UUID

from fastapi import APIRouter, Depends, status
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, selectinload

from app.api_errors import api_error
from app.database import get_database_session
from app.auth import get_current_user
from app.models import AnonymousIdentity, FaceImage, Person, User
from app.schemas import PersonCreate, PersonResponse
from app.vector_store import synchronize_face_id_safely


router = APIRouter(prefix="/api/anonymous-identities", tags=["anonymous identities"])


@router.post(
    "/{face_id}/enroll",
    response_model=PersonResponse,
    status_code=status.HTTP_201_CREATED,
)
def enroll_anonymous_identity(
    face_id: UUID,
    payload: PersonCreate,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_database_session),
) -> PersonResponse:
    identity = session.scalar(
        select(AnonymousIdentity)
        .options(selectinload(AnonymousIdentity.embeddings))
        .where(AnonymousIdentity.face_id == face_id)
        .with_for_update()
    )
    if identity is None:
        raise api_error(
            404,
            "ANONYMOUS_IDENTITY_NOT_FOUND",
            "Anonim yuz kimligi bulunamadi.",
        )
    if user.role != "admin" and identity.owner_user_id != user.id:
        raise api_error(404, "ANONYMOUS_IDENTITY_NOT_FOUND", "Anonim yuz kimligi bulunamadi.")
    if identity.person_id is not None:
        raise api_error(
            status.HTTP_409_CONFLICT,
            "IDENTITY_ALREADY_ENROLLED",
            "Bu anonim yuz daha once isimlendirilmis.",
        )
    if session.scalar(select(Person.id).where(Person.face_id == face_id)) is not None:
        raise api_error(
            status.HTTP_409_CONFLICT,
            "FACE_ID_ALREADY_REGISTERED",
            "Bu face ID zaten kayitli bir kisiye ait.",
        )

    sample_image_path = next(
        (sample.image_path for sample in identity.embeddings if sample.image_path),
        None,
    )
    person = Person(
        face_id=identity.face_id,
        owner_user_id=user.id,
        is_global=user.role == "admin",
        **payload.model_dump(),
    )
    try:
        session.add(person)
        session.flush()
        identity.person_id = person.id
        converted_count = 0
        for sample in list(identity.embeddings):
            if sample.image_path is None:
                continue
            session.add(
                FaceImage(
                    person_id=person.id,
                    image_path=sample.image_path,
                    embedding=sample.embedding,
                    detection_confidence=sample.detection_confidence,
                )
            )
            session.delete(sample)
            converted_count += 1
        session.commit()
        session.refresh(person)
    except SQLAlchemyError as error:
        session.rollback()
        raise api_error(
            500,
            "ANONYMOUS_ENROLL_FAILED",
            "Anonim yuz isimlendirilemedi.",
        ) from error

    synchronize_face_id_safely(session, face_id)

    return PersonResponse(
        id=person.id,
        face_id=person.face_id,
        first_name=person.first_name,
        last_name=person.last_name,
        description=person.description,
        face_image_count=converted_count,
        sample_image_url=f"/media/{sample_image_path}" if sample_image_path else None,
        created_at=person.created_at,
        updated_at=person.updated_at,
    )
