import logging
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Literal, Optional, Tuple
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Response, status
from sqlalchemy import func, or_, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, selectinload

from app.api_errors import api_error
from app.auth import get_current_user
from app.database import get_database_session
from app.face_storage import (
    finalize_staged_files,
    image_url,
    restore_staged_files,
    stage_face_images_for_deletion,
)
from app.models import (
    AnonymousFaceEmbedding,
    AnonymousIdentity,
    FaceImage,
    Person,
    RecognitionEvent,
    RecognitionProcess,
    User,
    VideoTrack,
)
from app.schemas import (
    FaceHistoryEntryResponse,
    FaceHistoryResponse,
    IdentityResponse,
    PersonUpdate,
)
from app.vector_store import synchronize_face_id_safely


router = APIRouter(prefix="/api/identities", tags=["identities"])
logger = logging.getLogger(__name__)
ObservationStats = Tuple[int, datetime]


def _person_by_face_id(session: Session, face_id: UUID) -> Optional[Person]:
    return session.scalar(
        select(Person)
        .options(selectinload(Person.face_images))
        .where(Person.face_id == face_id)
    )


def _anonymous_by_face_id(
    session: Session,
    face_id: UUID,
) -> Optional[AnonymousIdentity]:
    return session.scalar(
        select(AnonymousIdentity)
        .options(selectinload(AnonymousIdentity.embeddings))
        .where(AnonymousIdentity.face_id == face_id)
    )


def _observation_stats(
    session: Session,
    face_ids: Optional[List[UUID]] = None,
    owner_user_id: Optional[UUID] = None,
) -> Dict[UUID, ObservationStats]:
    if face_ids == []:
        return {}

    query = (
        select(
            RecognitionEvent.face_id,
            func.count(RecognitionEvent.id),
            func.max(RecognitionEvent.created_at),
        )
        .where(RecognitionEvent.face_id.is_not(None))
        .group_by(RecognitionEvent.face_id)
    )
    if face_ids is not None:
        query = query.where(RecognitionEvent.face_id.in_(face_ids))
    if owner_user_id is not None:
        query = query.join(
            RecognitionProcess,
            RecognitionProcess.process_id == RecognitionEvent.process_id,
        ).where(RecognitionProcess.owner_user_id == owner_user_id)

    return {
        face_id: (int(count), last_seen_at)
        for face_id, count, last_seen_at in session.execute(query)
        if face_id is not None and last_seen_at is not None
    }


def _video_observation_stats(
    session: Session,
    face_ids: Optional[List[UUID]] = None,
    owner_user_id: Optional[UUID] = None,
) -> Dict[UUID, ObservationStats]:
    if face_ids == []:
        return {}
    query = (
        select(
            VideoTrack.face_id,
            func.count(func.distinct(VideoTrack.process_id)),
            func.max(RecognitionProcess.created_at),
        )
        .join(
            RecognitionProcess,
            RecognitionProcess.process_id == VideoTrack.process_id,
        )
        .where(VideoTrack.face_id.is_not(None))
        .group_by(VideoTrack.face_id)
    )
    if face_ids is not None:
        query = query.where(VideoTrack.face_id.in_(face_ids))
    if owner_user_id is not None:
        query = query.where(RecognitionProcess.owner_user_id == owner_user_id)
    return {
        face_id: (int(count), last_seen_at)
        for face_id, count, last_seen_at in session.execute(query)
        if face_id is not None and last_seen_at is not None
    }


def _observed_face_ids(session: Session, owner_user_id: UUID) -> List[UUID]:
    photo_ids = session.scalars(
        select(RecognitionEvent.face_id)
        .join(
            RecognitionProcess,
            RecognitionProcess.process_id == RecognitionEvent.process_id,
        )
        .where(
            RecognitionProcess.owner_user_id == owner_user_id,
            RecognitionEvent.face_id.is_not(None),
        )
        .distinct()
    ).all()
    video_ids = session.scalars(
        select(VideoTrack.face_id)
        .join(
            RecognitionProcess,
            RecognitionProcess.process_id == VideoTrack.process_id,
        )
        .where(
            RecognitionProcess.owner_user_id == owner_user_id,
            VideoTrack.face_id.is_not(None),
        )
        .distinct()
    ).all()
    return list(dict.fromkeys([*photo_ids, *video_ids]))


def _identity_response(
    person: Optional[Person],
    anonymous: Optional[AnonymousIdentity],
    observation_stats: Optional[ObservationStats] = None,
    person_image_paths: Optional[List[str]] = None,
    anonymous_image_paths: Optional[List[str]] = None,
    video_observation_stats: Optional[ObservationStats] = None,
) -> IdentityResponse:
    if person is None and anonymous is None:
        raise ValueError("Identity data is required.")

    known = person is not None
    created_at = (
        anonymous.created_at
        if anonymous is not None
        else person.created_at  # type: ignore[union-attr]
    )
    updated_at = (
        person.updated_at
        if person is not None
        else anonymous.last_seen_at  # type: ignore[union-attr]
    )
    known_paths = (
        person_image_paths
        if person_image_paths is not None
        else [face.image_path for face in person.face_images] if person is not None else []
    )
    anonymous_paths = (
        anonymous_image_paths
        if anonymous_image_paths is not None
        else [
            sample.image_path
            for sample in anonymous.embeddings
            if sample.image_path is not None
        ] if anonymous is not None else []
    )
    reference_count = len(known_paths)
    observation_count = (
        observation_stats[0]
        if observation_stats is not None
        else anonymous.observation_count if anonymous is not None else 0
    )
    last_seen_at = (
        observation_stats[1]
        if observation_stats is not None
        else anonymous.last_seen_at if anonymous is not None else None
    )
    image_paths = [*known_paths, *anonymous_paths]
    return IdentityResponse(
        face_id=person.face_id if person is not None else anonymous.face_id,  # type: ignore[union-attr]
        status="known" if known else "anonymous",
        person_id=person.id if person is not None else None,
        first_name=person.first_name if person is not None else None,
        last_name=person.last_name if person is not None else None,
        description=person.description if person is not None else None,
        sample_count=reference_count + len(anonymous_paths),
        reference_image_count=reference_count,
        observation_count=observation_count,
        photo_observation_count=(
            observation_stats[0] if observation_stats is not None else 0
        ),
        photo_last_seen_at=(
            observation_stats[1] if observation_stats is not None else None
        ),
        video_observation_count=(
            video_observation_stats[0]
            if video_observation_stats is not None
            else 0
        ),
        video_last_seen_at=(
            video_observation_stats[1]
            if video_observation_stats is not None
            else None
        ),
        sample_image_urls=[image_url(path) for path in dict.fromkeys(image_paths)],
        created_at=created_at,
        updated_at=updated_at,
        last_seen_at=last_seen_at,
    )


@router.get("", response_model=List[IdentityResponse])
def list_identities(
    identity_status: Optional[Literal["known", "anonymous"]] = Query(
        default=None,
        alias="status",
    ),
    user: User = Depends(get_current_user),
    session: Session = Depends(get_database_session),
) -> List[IdentityResponse]:
    people_query = select(Person).order_by(Person.id)
    anonymous_query = select(AnonymousIdentity).order_by(AnonymousIdentity.id)
    if user.role != "admin":
        observed_face_ids = _observed_face_ids(session, user.id)
        people_query = people_query.where(
            or_(
                (Person.owner_user_id == user.id) & Person.is_global.is_(False),
                Person.face_id.in_(observed_face_ids),
            )
        )
        anonymous_query = anonymous_query.where(
            AnonymousIdentity.owner_user_id == user.id
        )
    people = session.scalars(people_query).all()
    anonymous_identities = session.scalars(anonymous_query).all()
    person_paths: Dict[int, List[str]] = defaultdict(list)
    for person_id, image_path in session.execute(
        select(FaceImage.person_id, FaceImage.image_path).order_by(FaceImage.id)
    ):
        person_paths[person_id].append(image_path)
    anonymous_paths: Dict[int, List[str]] = defaultdict(list)
    for identity_id, image_path in session.execute(
        select(
            AnonymousFaceEmbedding.anonymous_identity_id,
            AnonymousFaceEmbedding.image_path,
        ).order_by(AnonymousFaceEmbedding.id)
    ):
        if image_path is not None:
            anonymous_paths[identity_id].append(image_path)
    anonymous_by_face_id: Dict[UUID, AnonymousIdentity] = {
        identity.face_id: identity for identity in anonymous_identities
    }
    face_ids = [person.face_id for person in people]
    face_ids.extend(identity.face_id for identity in anonymous_identities)
    stats_by_face_id = _observation_stats(
        session,
        list(dict.fromkeys(face_ids)),
        None if user.role == "admin" else user.id,
    )
    video_stats_by_face_id = _video_observation_stats(
        session,
        list(dict.fromkeys(face_ids)),
        None if user.role == "admin" else user.id,
    )

    results: List[IdentityResponse] = []
    if identity_status in (None, "known"):
        results.extend(
            _identity_response(
                person,
                anonymous_by_face_id.get(person.face_id),
                stats_by_face_id.get(person.face_id),
                person_paths.get(person.id, []),
                (
                    anonymous_paths.get(anonymous_by_face_id[person.face_id].id, [])
                    if person.face_id in anonymous_by_face_id
                    else []
                ),
                video_stats_by_face_id.get(person.face_id),
            )
            for person in people
        )
    if identity_status in (None, "anonymous"):
        results.extend(
            _identity_response(
                None,
                identity,
                stats_by_face_id.get(identity.face_id),
                [],
                anonymous_paths.get(identity.id, []),
                video_stats_by_face_id.get(identity.face_id),
            )
            for identity in anonymous_identities
            if identity.person_id is None
        )
    return results


@router.get("/{face_id}", response_model=IdentityResponse)
def get_identity(
    face_id: UUID,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_database_session),
) -> IdentityResponse:
    person = _person_by_face_id(session, face_id)
    anonymous = _anonymous_by_face_id(session, face_id)
    if person is None and anonymous is None:
        raise api_error(404, "IDENTITY_NOT_FOUND", "Yuz kimligi bulunamadi.")
    if user.role != "admin" and not (
        (person is not None and person.owner_user_id == user.id and not person.is_global)
        or (anonymous is not None and anonymous.owner_user_id == user.id)
    ):
        raise api_error(404, "IDENTITY_NOT_FOUND", "Yuz kimligi bulunamadi.")
    stats = _observation_stats(
        session, [face_id], None if user.role == "admin" else user.id
    ).get(face_id)
    video_stats = _video_observation_stats(
        session, [face_id], None if user.role == "admin" else user.id
    ).get(face_id)
    return _identity_response(
        person,
        anonymous,
        stats,
        video_observation_stats=video_stats,
    )


@router.get("/{face_id}/history", response_model=FaceHistoryResponse)
def get_identity_history(
    face_id: UUID,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    user: User = Depends(get_current_user),
    session: Session = Depends(get_database_session),
) -> FaceHistoryResponse:
    person = _person_by_face_id(session, face_id)
    anonymous = _anonymous_by_face_id(session, face_id)
    if user.role != "admin" and not (
        (person is not None and (person.is_global or person.owner_user_id == user.id))
        or (anonymous is not None and anonymous.owner_user_id == user.id)
    ):
        raise api_error(404, "IDENTITY_NOT_FOUND", "Yuz kimligi bulunamadi.")
    event_filter = [RecognitionEvent.face_id == face_id]
    if user.role != "admin":
        event_filter.append(RecognitionProcess.owner_user_id == user.id)
    total, first_seen_at, last_seen_at = session.execute(
        select(
            func.count(RecognitionEvent.id),
            func.min(RecognitionEvent.created_at),
            func.max(RecognitionEvent.created_at),
        )
        .outerjoin(RecognitionProcess, RecognitionProcess.process_id == RecognitionEvent.process_id)
        .where(*event_filter)
    ).one()

    identity_exists = (
        session.scalar(select(Person.id).where(Person.face_id == face_id)) is not None
        or session.scalar(
            select(AnonymousIdentity.id).where(AnonymousIdentity.face_id == face_id)
        )
        is not None
    )
    if not identity_exists and total == 0:
        raise api_error(404, "IDENTITY_NOT_FOUND", "Yuz kimligi bulunamadi.")

    rows = session.execute(
        select(RecognitionEvent, RecognitionProcess.operation_type)
        .outerjoin(
            RecognitionProcess,
            RecognitionProcess.process_id == RecognitionEvent.process_id,
        )
        .where(*event_filter)
        .order_by(RecognitionEvent.created_at.desc(), RecognitionEvent.id.desc())
        .offset(offset)
        .limit(limit)
    ).all()

    return FaceHistoryResponse(
        face_id=face_id,
        total=int(total),
        limit=limit,
        offset=offset,
        first_seen_at=first_seen_at,
        last_seen_at=last_seen_at,
        appearances=[
            FaceHistoryEntryResponse(
                event_id=event.id,
                process_id=event.process_id,
                operation_type=operation_type,
                timestamp=event.created_at,
                status=event.face_status,
                recognized=event.recognized,
                similarity=event.similarity,
                threshold=event.threshold,
            )
            for event, operation_type in rows
        ],
    )


@router.patch("/{face_id}", response_model=IdentityResponse)
def update_identity(
    face_id: UUID,
    payload: PersonUpdate,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_database_session),
) -> IdentityResponse:
    person = _person_by_face_id(session, face_id)
    if person is None:
        if _anonymous_by_face_id(session, face_id) is not None:
            raise api_error(
                status.HTTP_409_CONFLICT,
                "ANONYMOUS_IDENTITY_REQUIRES_ENROLLMENT",
                "Anonim kimlik guncellenmeden once isimlendirilmelidir.",
            )
        raise api_error(404, "IDENTITY_NOT_FOUND", "Yuz kimligi bulunamadi.")
    if user.role != "admin" and (person.owner_user_id != user.id or person.is_global):
        raise api_error(404, "IDENTITY_NOT_FOUND", "Yuz kimligi bulunamadi.")

    for field_name, value in payload.model_dump(exclude_unset=True).items():
        setattr(person, field_name, value)
    try:
        session.commit()
    except SQLAlchemyError as error:
        session.rollback()
        raise api_error(
            500,
            "IDENTITY_UPDATE_FAILED",
            "Yuz kimligi guncellenemedi.",
        ) from error

    return _identity_response(
        _person_by_face_id(session, face_id),
        _anonymous_by_face_id(session, face_id),
        _observation_stats(session, [face_id]).get(face_id),
    )


@router.delete("/{face_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_identity(
    face_id: UUID,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_database_session),
) -> Response:
    person = _person_by_face_id(session, face_id)
    anonymous = _anonymous_by_face_id(session, face_id)
    if person is None and anonymous is None:
        raise api_error(404, "IDENTITY_NOT_FOUND", "Yuz kimligi bulunamadi.")
    if user.role != "admin" and not (
        (person is not None and person.owner_user_id == user.id and not person.is_global)
        or (anonymous is not None and anonymous.owner_user_id == user.id)
    ):
        raise api_error(404, "IDENTITY_NOT_FOUND", "Yuz kimligi bulunamadi.")

    try:
        image_paths = [
            face.image_path for face in (person.face_images if person is not None else [])
        ]
        image_paths.extend(
            sample.image_path
            for sample in (anonymous.embeddings if anonymous is not None else [])
            if sample.image_path is not None
        )
        staged_files = stage_face_images_for_deletion(image_paths)
    except (OSError, ValueError) as error:
        raise api_error(
            500,
            "FACE_FILE_STAGE_FAILED",
            "Yuz dosyalari silmeye hazirlanamadi.",
        ) from error

    try:
        if anonymous is not None:
            session.delete(anonymous)
        if person is not None:
            session.delete(person)
        session.commit()
    except SQLAlchemyError as error:
        session.rollback()
        restore_staged_files(staged_files)
        raise api_error(
            500,
            "IDENTITY_DELETE_FAILED",
            "Yuz kimligi silinemedi.",
        ) from error

    synchronize_face_id_safely(session, face_id)

    try:
        finalize_staged_files(staged_files)
    except OSError:
        logger.exception("Staged files could not be finalized for face ID %s", face_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
