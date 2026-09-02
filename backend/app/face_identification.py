import logging
import os
from typing import List, Optional, Tuple
from uuid import UUID

from sqlalchemy import func, or_, select, text
from sqlalchemy.orm import Session

from app.models import (
    AnonymousFaceEmbedding,
    AnonymousIdentity,
    FaceImage,
    Person,
)
from app.vector_store import VectorStoreUnavailable, search_vectors


IdentificationMatch = Tuple[Person, Optional[FaceImage], float]
AnonymousIdentificationMatch = Tuple[AnonymousIdentity, float]
AnonymousIdentityCreation = Tuple[AnonymousIdentity, AnonymousFaceEmbedding]
ANONYMOUS_MATCH_LOCK_ID = 716238941
ANONYMOUS_FACE_MAX_SAMPLES = int(os.getenv("ANONYMOUS_FACE_MAX_SAMPLES", "5"))
ANONYMOUS_FACE_DUPLICATE_THRESHOLD = float(
    os.getenv("ANONYMOUS_FACE_DUPLICATE_THRESHOLD", "0.98")
)
logger = logging.getLogger(__name__)


def find_closest_face(
    session: Session,
    embedding: List[float],
    exclude_person_id: Optional[int] = None,
    owner_user_id: Optional[UUID] = None,
) -> Optional[IdentificationMatch]:
    try:
        qdrant_results = search_vectors(
            embedding,
            status="known",
            exclude_person_id=exclude_person_id,
            owner_user_id=owner_user_id,
        )
        for result in qdrant_results:
            sample_type = result.payload.get("sampleType")
            sample_id = result.payload.get("sampleId")
            if not isinstance(sample_id, int):
                continue

            if sample_type == "face_image":
                row = session.execute(
                    select(Person, FaceImage)
                    .join(FaceImage, FaceImage.person_id == Person.id)
                    .where(FaceImage.id == sample_id)
                ).first()
                if row is None:
                    continue
                person, face_image = row
                if exclude_person_id is not None and person.id == exclude_person_id:
                    continue
                return person, face_image, result.score

            if sample_type == "anonymous_embedding":
                person = session.scalar(
                    select(Person)
                    .join(AnonymousIdentity, AnonymousIdentity.person_id == Person.id)
                    .join(
                        AnonymousFaceEmbedding,
                        AnonymousFaceEmbedding.anonymous_identity_id
                        == AnonymousIdentity.id,
                    )
                    .where(AnonymousFaceEmbedding.id == sample_id)
                )
                if person is None:
                    continue
                if exclude_person_id is not None and person.id == exclude_person_id:
                    continue
                return person, None, result.score
    except VectorStoreUnavailable:
        logger.warning("Qdrant unavailable during known search; using pgvector fallback")

    return _find_closest_face_postgres(
        session, embedding, exclude_person_id, owner_user_id
    )


def _find_closest_face_postgres(
    session: Session,
    embedding: List[float],
    exclude_person_id: Optional[int] = None,
    owner_user_id: Optional[UUID] = None,
) -> Optional[IdentificationMatch]:
    image_distance = FaceImage.embedding.cosine_distance(embedding).label("distance")
    image_query = (
        select(Person, FaceImage, image_distance)
        .join(FaceImage, FaceImage.person_id == Person.id)
    )
    if exclude_person_id is not None:
        image_query = image_query.where(Person.id != exclude_person_id)
    if owner_user_id is not None:
        image_query = image_query.where(
            or_(Person.is_global.is_(True), Person.owner_user_id == owner_user_id)
        )
    image_result = session.execute(
        image_query.order_by(image_distance).limit(1)
    ).first()

    enrolled_distance = AnonymousFaceEmbedding.embedding.cosine_distance(
        embedding
    ).label("distance")
    enrolled_query = (
        select(Person, enrolled_distance)
        .join(AnonymousIdentity, AnonymousIdentity.person_id == Person.id)
        .join(
            AnonymousFaceEmbedding,
            AnonymousFaceEmbedding.anonymous_identity_id == AnonymousIdentity.id,
        )
        .where(AnonymousIdentity.person_id.is_not(None))
    )
    if exclude_person_id is not None:
        enrolled_query = enrolled_query.where(Person.id != exclude_person_id)
    if owner_user_id is not None:
        enrolled_query = enrolled_query.where(
            or_(Person.is_global.is_(True), Person.owner_user_id == owner_user_id)
        )
    enrolled_result = session.execute(
        enrolled_query.order_by(enrolled_distance).limit(1)
    ).first()

    candidates: List[IdentificationMatch] = []
    if image_result is not None:
        person, face_image, cosine_distance = image_result
        candidates.append(
            (person, face_image, _similarity_from_distance(cosine_distance))
        )
    if enrolled_result is not None:
        person, cosine_distance = enrolled_result
        candidates.append((person, None, _similarity_from_distance(cosine_distance)))
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: candidate[2])


def _similarity_from_distance(cosine_distance: float) -> float:
    similarity = 1.0 - float(cosine_distance)
    return max(-1.0, min(1.0, similarity))


def lock_anonymous_matching(session: Session) -> None:
    session.execute(
        text("SELECT pg_advisory_xact_lock(:lock_id)"),
        {"lock_id": ANONYMOUS_MATCH_LOCK_ID},
    )


def find_closest_anonymous_face(
    session: Session,
    embedding: List[float],
    owner_user_id: Optional[UUID] = None,
) -> Optional[AnonymousIdentificationMatch]:
    try:
        qdrant_results = search_vectors(
            embedding, status="anonymous", owner_user_id=owner_user_id
        )
        for result in qdrant_results:
            if result.payload.get("sampleType") != "anonymous_embedding":
                continue
            sample_id = result.payload.get("sampleId")
            if not isinstance(sample_id, int):
                continue
            identity = session.scalar(
                select(AnonymousIdentity)
                .join(
                    AnonymousFaceEmbedding,
                    AnonymousFaceEmbedding.anonymous_identity_id
                    == AnonymousIdentity.id,
                )
                .where(
                    AnonymousFaceEmbedding.id == sample_id,
                    AnonymousIdentity.person_id.is_(None),
                )
            )
            if identity is not None:
                if owner_user_id is not None and identity.owner_user_id != owner_user_id:
                    continue
                return identity, result.score
    except VectorStoreUnavailable:
        logger.warning("Qdrant unavailable during anonymous search; using pgvector fallback")

    return _find_closest_anonymous_face_postgres(
        session, embedding, owner_user_id
    )


def _find_closest_anonymous_face_postgres(
    session: Session,
    embedding: List[float],
    owner_user_id: Optional[UUID] = None,
) -> Optional[AnonymousIdentificationMatch]:
    distance = AnonymousFaceEmbedding.embedding.cosine_distance(embedding).label(
        "distance"
    )
    query = (
        select(AnonymousIdentity, distance)
        .join(
            AnonymousFaceEmbedding,
            AnonymousFaceEmbedding.anonymous_identity_id == AnonymousIdentity.id,
        )
        .where(AnonymousIdentity.person_id.is_(None))
    )
    if owner_user_id is not None:
        query = query.where(AnonymousIdentity.owner_user_id == owner_user_id)
    query = query.order_by(distance).limit(1)
    result = session.execute(query).first()
    if result is None:
        return None

    identity, cosine_distance = result
    return identity, _similarity_from_distance(cosine_distance)


def create_anonymous_identity(
    session: Session,
    embedding: List[float],
    detection_confidence: float,
    owner_user_id: Optional[UUID] = None,
) -> AnonymousIdentityCreation:
    identity = AnonymousIdentity(owner_user_id=owner_user_id)
    session.add(identity)
    session.flush()
    sample = AnonymousFaceEmbedding(
        anonymous_identity_id=identity.id,
        embedding=embedding,
        detection_confidence=detection_confidence,
    )
    session.add(sample)
    return identity, sample


def record_anonymous_observation(
    session: Session,
    identity: AnonymousIdentity,
    embedding: List[float],
    detection_confidence: float,
    match_similarity: float,
) -> Optional[AnonymousFaceEmbedding]:
    identity.observation_count += 1
    identity.last_seen_at = func.now()
    if (
        len(identity.embeddings) >= ANONYMOUS_FACE_MAX_SAMPLES
        or match_similarity >= ANONYMOUS_FACE_DUPLICATE_THRESHOLD
    ):
        return None

    sample = AnonymousFaceEmbedding(
        anonymous_identity_id=identity.id,
        embedding=embedding,
        detection_confidence=detection_confidence,
    )
    session.add(sample)
    return sample
