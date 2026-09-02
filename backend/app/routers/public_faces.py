from typing import Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, Form, Response, UploadFile, status
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, selectinload

from app.api_errors import api_error
from app.auth import get_current_user
from app.database import get_database_session
from app.face_detector import DUPLICATE_FACE_THRESHOLD
from app.face_identification import find_closest_face
from app.face_storage import delete_face_image, save_face_image
from app.image_upload import extract_face_data_or_error, read_uploaded_image
from app.models import AnonymousIdentity, FaceImage, Person, User
from app.routers.identities import (
    _anonymous_by_face_id,
    _identity_response,
    _observation_stats,
    _person_by_face_id,
    delete_identity,
    get_identity_history,
)
from app.schemas import FaceHistoryResponse, IdentityResponse
from app.vector_store import synchronize_face_id_safely


router = APIRouter(prefix="/faces", tags=["faces"])


def _required_name(value: Optional[str], field_name: str) -> str:
    normalized = (value or "").strip()
    if not normalized:
        raise api_error(
            422,
            "VALIDATION_ERROR",
            f"{field_name} bos birakilamaz.",
            {"field": field_name},
        )
    if len(normalized) > 100:
        raise api_error(
            422,
            "VALIDATION_ERROR",
            f"{field_name} en fazla 100 karakter olabilir.",
            {"field": field_name, "max_length": 100},
        )
    return normalized


def _optional_description(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = value.strip()
    if len(normalized) > 1000:
        raise api_error(
            422,
            "VALIDATION_ERROR",
            "description en fazla 1000 karakter olabilir.",
            {"field": "description", "max_length": 1000},
        )
    return normalized or None


def _response_for(session: Session, face_id: UUID, user: User) -> IdentityResponse:
    person = _person_by_face_id(session, face_id)
    anonymous = _anonymous_by_face_id(session, face_id)
    return _identity_response(
        person,
        anonymous,
        _observation_stats(
            session,
            [face_id],
            None if user.role == "admin" else user.id,
        ).get(face_id),
    )


@router.post(
    "/enroll",
    response_model=IdentityResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Enroll a face",
)
async def enroll_face(
    face_id: Optional[UUID] = Form(
        default=None,
        alias="faceId",
        description="Isimlendirilecek anonim veya guncellenecek kayitli face ID.",
    ),
    first_name: Optional[str] = Form(default=None, max_length=100),
    last_name: Optional[str] = Form(default=None, max_length=100),
    description: Optional[str] = Form(default=None, max_length=1000),
    file: Optional[UploadFile] = File(
        default=None,
        description="Yeni kayit veya ek yuz ornegi icin JPEG, PNG ya da WebP.",
    ),
    user: User = Depends(get_current_user),
    session: Session = Depends(get_database_session),
) -> IdentityResponse:
    person = _person_by_face_id(session, face_id) if face_id is not None else None
    anonymous = (
        session.scalar(
            select(AnonymousIdentity)
            .options(selectinload(AnonymousIdentity.embeddings))
            .where(AnonymousIdentity.face_id == face_id)
            .with_for_update()
        )
        if face_id is not None
        else None
    )
    if face_id is not None and person is None and anonymous is None:
        raise api_error(404, "IDENTITY_NOT_FOUND", "Yuz kimligi bulunamadi.")
    if user.role != "admin" and (
        (person is not None and (person.owner_user_id != user.id or person.is_global))
        or (anonymous is not None and anonymous.owner_user_id != user.id)
    ):
        raise api_error(404, "IDENTITY_NOT_FOUND", "Yuz kimligi bulunamadi.")

    image = None
    embedding = None
    confidence = None
    if file is not None:
        image = await read_uploaded_image(file, "file")
        detected_embedding, confidence = extract_face_data_or_error(image, "file")
        embedding = detected_embedding.tolist()

        duplicate = find_closest_face(
            session,
            embedding,
            exclude_person_id=person.id if person is not None else None,
            owner_user_id=user.id,
        )
        if duplicate is not None and duplicate[2] >= DUPLICATE_FACE_THRESHOLD:
            matched_person, _, similarity = duplicate
            raise api_error(
                409,
                "DUPLICATE_FACE",
                "Bu yuz baska bir kayitli kimlige ait.",
                {
                    "matched_face_id": str(matched_person.face_id),
                    "similarity": round(similarity, 4),
                },
            )

    stored_path = None
    try:
        if person is None:
            normalized_first_name = _required_name(first_name, "first_name")
            normalized_last_name = _required_name(last_name, "last_name")
            if face_id is None and image is None:
                raise api_error(
                    422,
                    "FACE_IMAGE_REQUIRED",
                    "Yeni bir kimlik kaydetmek icin yuz fotografi gereklidir.",
                )

            person = Person(
                face_id=anonymous.face_id if anonymous is not None else uuid4(),
                first_name=normalized_first_name,
                last_name=normalized_last_name,
                description=_optional_description(description),
                owner_user_id=user.id,
                is_global=user.role == "admin",
            )
            session.add(person)
            session.flush()

            if anonymous is not None:
                anonymous.person_id = person.id
                for sample in list(anonymous.embeddings):
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
        else:
            if first_name is not None:
                person.first_name = _required_name(first_name, "first_name")
            if last_name is not None:
                person.last_name = _required_name(last_name, "last_name")
            if description is not None:
                person.description = _optional_description(description)
            if file is None and all(
                value is None for value in (first_name, last_name, description)
            ):
                raise api_error(
                    422,
                    "ENROLLMENT_DATA_REQUIRED",
                    "En az bir fotograf veya metadata alani gonderilmelidir.",
                )

        if image is not None and embedding is not None and confidence is not None:
            stored_path = save_face_image(person.id, image)
            session.add(
                FaceImage(
                    person_id=person.id,
                    image_path=stored_path,
                    embedding=embedding,
                    detection_confidence=confidence,
                )
            )

        session.commit()
    except SQLAlchemyError as error:
        session.rollback()
        if stored_path is not None:
            delete_face_image(stored_path)
        raise api_error(
            500,
            "FACE_ENROLL_FAILED",
            "Yuz kimligi kaydedilemedi.",
        ) from error
    except Exception:
        session.rollback()
        if stored_path is not None:
            delete_face_image(stored_path)
        raise

    synchronize_face_id_safely(session, person.face_id)
    return _response_for(session, person.face_id, user)


@router.get(
    "/{faceId}",
    response_model=IdentityResponse,
    summary="Get face details",
)
def get_face(
    faceId: UUID,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_database_session),
) -> IdentityResponse:
    person = _person_by_face_id(session, faceId)
    anonymous = _anonymous_by_face_id(session, faceId)
    if person is None and anonymous is None:
        raise api_error(404, "IDENTITY_NOT_FOUND", "Yuz kimligi bulunamadi.")
    if user.role != "admin" and not (
        (person is not None and (person.is_global or person.owner_user_id == user.id))
        or (anonymous is not None and anonymous.owner_user_id == user.id)
    ):
        raise api_error(404, "IDENTITY_NOT_FOUND", "Yuz kimligi bulunamadi.")
    return _response_for(session, faceId, user)


@router.delete(
    "/{faceId}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a face",
)
def delete_face(
    faceId: UUID,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_database_session),
) -> Response:
    return delete_identity(face_id=faceId, user=user, session=session)


@router.get(
    "/{faceId}/history",
    response_model=FaceHistoryResponse,
    summary="Get face history",
)
def get_face_history(
    faceId: UUID,
    limit: int = 20,
    offset: int = 0,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_database_session),
) -> FaceHistoryResponse:
    if not 1 <= limit <= 100 or offset < 0:
        raise api_error(
            422,
            "VALIDATION_ERROR",
            "limit 1-100 arasinda, offset ise sifir veya daha buyuk olmalidir.",
        )
    return get_identity_history(
        face_id=faceId,
        limit=limit,
        offset=offset,
        user=user,
        session=session,
    )
