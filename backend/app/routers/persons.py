from typing import List, Optional

import logging

from fastapi import (
    APIRouter,
    Depends,
    File,
    Response,
    UploadFile,
    status,
)
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, selectinload

from app.api_errors import api_error
from app.database import get_database_session
from app.face_detector import DUPLICATE_FACE_THRESHOLD, active_execution_providers
from app.face_identification import find_closest_face
from app.face_storage import (
    delete_face_image,
    finalize_staged_files,
    image_url,
    restore_staged_files,
    save_face_image,
    stage_face_images_for_deletion,
)
from app.image_upload import extract_face_data_or_error, read_uploaded_image
from app.models import AnonymousIdentity, FaceImage, Person
from app.schemas import (
    FaceImageResponse,
    FaceImageUploadResponse,
    PersonCreate,
    PersonResponse,
    PersonUpdate,
)
from app.vector_store import synchronize_face_id_safely


router = APIRouter(prefix="/api/persons", tags=["persons"])
logger = logging.getLogger(__name__)


def get_person_with_faces(session: Session, person_id: int) -> Person:
    query = (
        select(Person)
        .options(selectinload(Person.face_images))
        .where(Person.id == person_id)
    )
    person = session.scalar(query)
    if person is None:
        raise api_error(404, "PERSON_NOT_FOUND", "Kisi bulunamadi.")
    return person


def person_response(
    person: Person,
    face_image_count: Optional[int] = None,
    sample_image_path: Optional[str] = None,
) -> PersonResponse:
    if face_image_count is None:
        sample_image_path = (
            person.face_images[0].image_path if person.face_images else None
        )
    return PersonResponse(
        id=person.id,
        face_id=person.face_id,
        first_name=person.first_name,
        last_name=person.last_name,
        description=person.description,
        face_image_count=(
            face_image_count
            if face_image_count is not None
            else len(person.face_images)
        ),
        sample_image_url=(
            image_url(sample_image_path) if sample_image_path is not None else None
        ),
        created_at=person.created_at,
        updated_at=person.updated_at,
    )


def face_image_response(face_image: FaceImage) -> FaceImageResponse:
    return FaceImageResponse(
        id=face_image.id,
        person_id=face_image.person_id,
        image_url=image_url(face_image.image_path),
        detection_confidence=round(face_image.detection_confidence, 4),
        created_at=face_image.created_at,
    )


@router.post("", response_model=PersonResponse, status_code=status.HTTP_201_CREATED)
def create_person(
    payload: PersonCreate,
    session: Session = Depends(get_database_session),
) -> PersonResponse:
    person = Person(**payload.model_dump(), is_global=True)
    session.add(person)
    session.commit()
    session.refresh(person)
    return person_response(person)


@router.get("", response_model=List[PersonResponse])
def list_persons(
    session: Session = Depends(get_database_session),
) -> List[PersonResponse]:
    query = (
        select(Person, func.count(FaceImage.id), func.min(FaceImage.image_path))
        .outerjoin(FaceImage, FaceImage.person_id == Person.id)
        .group_by(Person.id)
        .order_by(Person.id)
    )
    return [
        person_response(person, int(face_image_count), sample_image_path)
        for person, face_image_count, sample_image_path in session.execute(query)
    ]


@router.get("/{person_id}", response_model=PersonResponse)
def get_person(
    person_id: int,
    session: Session = Depends(get_database_session),
) -> PersonResponse:
    return person_response(get_person_with_faces(session, person_id))


@router.patch("/{person_id}", response_model=PersonResponse)
def update_person(
    person_id: int,
    payload: PersonUpdate,
    session: Session = Depends(get_database_session),
) -> PersonResponse:
    person = get_person_with_faces(session, person_id)
    for field_name, value in payload.model_dump(exclude_unset=True).items():
        setattr(person, field_name, value)

    try:
        session.commit()
    except SQLAlchemyError:
        session.rollback()
        raise api_error(500, "PERSON_UPDATE_FAILED", "Kisi guncellenemedi.")

    return person_response(get_person_with_faces(session, person_id))


@router.delete("/{person_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_person(
    person_id: int,
    session: Session = Depends(get_database_session),
) -> Response:
    person = get_person_with_faces(session, person_id)
    face_id = person.face_id
    enrolled_identity = session.scalar(
        select(AnonymousIdentity)
        .options(selectinload(AnonymousIdentity.embeddings))
        .where(AnonymousIdentity.person_id == person.id)
    )
    image_paths = [face_image.image_path for face_image in person.face_images]
    if enrolled_identity is not None:
        image_paths.extend(
            sample.image_path
            for sample in enrolled_identity.embeddings
            if sample.image_path is not None
        )
    try:
        staged_files = stage_face_images_for_deletion(image_paths)
    except (OSError, ValueError):
        raise api_error(
            500,
            "FACE_FILE_STAGE_FAILED",
            "Yuz dosyalari silmeye hazirlanamadi.",
        )

    try:
        if enrolled_identity is not None:
            session.delete(enrolled_identity)
        session.delete(person)
        session.commit()
    except SQLAlchemyError:
        session.rollback()
        restore_staged_files(staged_files)
        raise api_error(500, "PERSON_DELETE_FAILED", "Kisi silinemedi.")

    synchronize_face_id_safely(session, face_id)

    try:
        finalize_staged_files(staged_files)
    except OSError:
        logger.exception("Staged face images could not be finalized for person %s", person_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/{person_id}/face-images",
    response_model=FaceImageUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_face_image(
    person_id: int,
    file: UploadFile = File(
        ...,
        description="Tek yuz iceren JPEG, PNG veya WebP goruntu; en fazla 10 MB.",
    ),
    session: Session = Depends(get_database_session),
) -> FaceImageUploadResponse:
    person = session.get(Person, person_id)
    if person is None:
        raise api_error(404, "PERSON_NOT_FOUND", "Kisi bulunamadi.")

    image = await read_uploaded_image(file, "file")
    embedding, confidence = extract_face_data_or_error(image, "file")

    duplicate_match = find_closest_face(
        session,
        embedding.tolist(),
        exclude_person_id=person_id,
    )
    if duplicate_match is not None:
        matched_person, _, similarity = duplicate_match
        if similarity >= DUPLICATE_FACE_THRESHOLD:
            raise api_error(
                status.HTTP_409_CONFLICT,
                "DUPLICATE_FACE",
                (
                    "Bu yüz zaten "
                    f"{matched_person.first_name} {matched_person.last_name} "
                    f"adlı kişiye kayıtlı (benzerlik: %{similarity * 100:.2f})."
                ),
                {
                    "matched_person_id": matched_person.id,
                    "similarity": round(similarity, 4),
                },
            )

    stored_path = save_face_image(person_id, image)

    face_image = FaceImage(
        person_id=person_id,
        image_path=stored_path,
        embedding=embedding.tolist(),
        detection_confidence=confidence,
    )
    try:
        session.add(face_image)
        session.commit()
        session.refresh(face_image)
    except SQLAlchemyError:
        session.rollback()
        delete_face_image(stored_path)
        raise api_error(
            500,
            "FACE_DATABASE_WRITE_FAILED",
            "Yuz kaydi veritabanina yazilamadi.",
        )

    synchronize_face_id_safely(session, person.face_id)

    response = face_image_response(face_image)
    return FaceImageUploadResponse(
        **response.model_dump(),
        execution_providers=active_execution_providers(),
    )


@router.get(
    "/{person_id}/face-images",
    response_model=List[FaceImageResponse],
)
def list_face_images(
    person_id: int,
    session: Session = Depends(get_database_session),
) -> List[FaceImageResponse]:
    if session.get(Person, person_id) is None:
        raise api_error(404, "PERSON_NOT_FOUND", "Kisi bulunamadi.")

    query = (
        select(FaceImage)
        .where(FaceImage.person_id == person_id)
        .order_by(FaceImage.id)
    )
    return [face_image_response(item) for item in session.scalars(query).all()]


@router.delete(
    "/{person_id}/face-images/{face_image_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_person_face_image(
    person_id: int,
    face_image_id: int,
    session: Session = Depends(get_database_session),
) -> Response:
    person = session.get(Person, person_id)
    if person is None:
        raise api_error(404, "PERSON_NOT_FOUND", "Kisi bulunamadi.")

    query = select(FaceImage).where(
        FaceImage.id == face_image_id,
        FaceImage.person_id == person_id,
    )
    face_image = session.scalar(query)
    if face_image is None:
        raise api_error(404, "FACE_IMAGE_NOT_FOUND", "Yuz fotografi bulunamadi.")

    try:
        staged_files = stage_face_images_for_deletion([face_image.image_path])
    except (OSError, ValueError):
        raise api_error(
            500,
            "FACE_FILE_STAGE_FAILED",
            "Yuz dosyasi silmeye hazirlanamadi.",
        )

    try:
        session.delete(face_image)
        session.commit()
    except SQLAlchemyError:
        session.rollback()
        restore_staged_files(staged_files)
        raise api_error(500, "FACE_IMAGE_DELETE_FAILED", "Yuz fotografi silinemedi.")

    synchronize_face_id_safely(session, person.face_id)

    try:
        finalize_staged_files(staged_files)
    except OSError:
        logger.exception(
            "Staged face image could not be finalized for face image %s", face_image_id
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)
