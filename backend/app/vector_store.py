import logging
import os
from dataclasses import dataclass
from threading import Lock
from typing import Dict, Iterable, Iterator, List, Optional, Set
from uuid import UUID, uuid5

from qdrant_client import QdrantClient, models
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import AnonymousFaceEmbedding, AnonymousIdentity, FaceImage, Person


logger = logging.getLogger(__name__)
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "face_embeddings")
QDRANT_PREFER_GRPC = os.getenv("QDRANT_PREFER_GRPC", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
FACE_VECTOR_SIZE = int(os.getenv("FACE_VECTOR_SIZE", "512"))
QDRANT_UPSERT_BATCH_SIZE = int(os.getenv("QDRANT_UPSERT_BATCH_SIZE", "256"))
QDRANT_INDEXING_THRESHOLD_KB = max(
    1, int(os.getenv("QDRANT_INDEXING_THRESHOLD_KB", "1000"))
)
QDRANT_FULL_SCAN_THRESHOLD_KB = max(
    1, int(os.getenv("QDRANT_FULL_SCAN_THRESHOLD_KB", "1000"))
)
FACE_EMBEDDING_MODEL_VERSION = os.getenv(
    "FACE_EMBEDDING_MODEL_VERSION", "arcface_r50_w600k_v1"
)
POINT_NAMESPACE = UUID("39cc9e99-17f3-4388-ae60-e8f0d5ee291d")
_client = QdrantClient(
    url=QDRANT_URL,
    timeout=5.0,
    prefer_grpc=QDRANT_PREFER_GRPC,
)
_collection_lock = Lock()
_collection_ready = False


class VectorStoreUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class VectorSearchResult:
    point_id: UUID
    score: float
    payload: Dict[str, object]


@dataclass(frozen=True)
class VectorSyncResult:
    indexed_points: int
    removed_points: int


def _point_id(sample_type: str, sample_id: int) -> UUID:
    return uuid5(POINT_NAMESPACE, f"{sample_type}:{sample_id}")


def _face_image_point(face_image: FaceImage, person: Person) -> models.PointStruct:
    return models.PointStruct(
        id=str(_point_id("face_image", face_image.id)),
        vector=list(face_image.embedding),
        payload={
            "faceId": str(person.face_id),
            "personId": person.id,
            "status": "known",
            "sampleType": "face_image",
            "sampleId": face_image.id,
            "imagePath": face_image.image_path,
            "modelVersion": FACE_EMBEDDING_MODEL_VERSION,
            "ownerId": str(person.owner_user_id) if person.owner_user_id else None,
            "isGlobal": person.is_global,
        },
    )


def _anonymous_point(
    sample: AnonymousFaceEmbedding,
    identity: AnonymousIdentity,
) -> models.PointStruct:
    return models.PointStruct(
        id=str(_point_id("anonymous_embedding", sample.id)),
        vector=list(sample.embedding),
        payload={
            "faceId": str(identity.face_id),
            "personId": identity.person_id,
            "status": "known" if identity.person_id is not None else "anonymous",
            "sampleType": "anonymous_embedding",
            "sampleId": sample.id,
            "imagePath": sample.image_path,
            "modelVersion": FACE_EMBEDDING_MODEL_VERSION,
            "ownerId": str(identity.owner_user_id) if identity.owner_user_id else None,
            "isGlobal": False,
        },
    )


def _collection_names() -> Set[str]:
    return {item.name for item in _client.get_collections().collections}


def _ensure_index_config(collection: object) -> None:
    optimizer_config = collection.config.optimizer_config
    hnsw_config = collection.config.hnsw_config
    if (
        optimizer_config.indexing_threshold == QDRANT_INDEXING_THRESHOLD_KB
        and hnsw_config.full_scan_threshold == QDRANT_FULL_SCAN_THRESHOLD_KB
    ):
        return

    _client.update_collection(
        collection_name=QDRANT_COLLECTION,
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=QDRANT_INDEXING_THRESHOLD_KB,
        ),
        hnsw_config=models.HnswConfigDiff(
            full_scan_threshold=QDRANT_FULL_SCAN_THRESHOLD_KB,
        ),
        timeout=60,
    )
    logger.info(
        "Qdrant HNSW configuration updated: indexing_threshold_kb=%s "
        "full_scan_threshold_kb=%s",
        QDRANT_INDEXING_THRESHOLD_KB,
        QDRANT_FULL_SCAN_THRESHOLD_KB,
    )


def ensure_collection() -> None:
    global _collection_ready
    if _collection_ready:
        return
    with _collection_lock:
        if _collection_ready:
            return
        if QDRANT_COLLECTION not in _collection_names():
            _client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=models.VectorParams(
                    size=FACE_VECTOR_SIZE,
                    distance=models.Distance.COSINE,
                ),
            )

        collection = _client.get_collection(QDRANT_COLLECTION)
        _ensure_index_config(collection)
        payload_schema = collection.payload_schema or {}
        field_schemas = {
            "faceId": models.PayloadSchemaType.KEYWORD,
            "status": models.PayloadSchemaType.KEYWORD,
            "personId": models.PayloadSchemaType.INTEGER,
            "modelVersion": models.PayloadSchemaType.KEYWORD,
            "ownerId": models.PayloadSchemaType.KEYWORD,
            "isGlobal": models.PayloadSchemaType.BOOL,
        }
        for field_name, field_schema in field_schemas.items():
            if field_name in payload_schema:
                continue
            _client.create_payload_index(
                collection_name=QDRANT_COLLECTION,
                field_name=field_name,
                field_schema=field_schema,
                wait=True,
            )
        _collection_ready = True


def _database_points(session: Session) -> Iterator[models.PointStruct]:
    face_images = select(FaceImage, Person).join(
        Person,
        Person.id == FaceImage.person_id,
    ).execution_options(yield_per=QDRANT_UPSERT_BATCH_SIZE)
    for face_image, person in session.execute(face_images):
        yield _face_image_point(face_image, person)

    anonymous_samples = select(
        AnonymousFaceEmbedding,
        AnonymousIdentity,
    ).join(
        AnonymousIdentity,
        AnonymousIdentity.id == AnonymousFaceEmbedding.anonymous_identity_id,
    ).execution_options(yield_per=QDRANT_UPSERT_BATCH_SIZE)
    for sample, identity in session.execute(anonymous_samples):
        yield _anonymous_point(sample, identity)


def _stored_point_ids(query_filter: Optional[models.Filter] = None) -> Set[str]:
    point_ids: Set[str] = set()
    offset = None
    while True:
        records, next_offset = _client.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=query_filter,
            limit=256,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        point_ids.update(str(record.id) for record in records)
        if next_offset is None:
            return point_ids
        offset = next_offset


def _upsert(points: Iterable[models.PointStruct]) -> int:
    indexed_points = 0
    batch: List[models.PointStruct] = []
    for point in points:
        batch.append(point)
        if len(batch) < QDRANT_UPSERT_BATCH_SIZE:
            continue
        _client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=batch,
            wait=True,
        )
        indexed_points += len(batch)
        batch = []

    if batch:
        _client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=batch,
            wait=True,
        )
        indexed_points += len(batch)
    return indexed_points


def synchronize_all(session: Session) -> VectorSyncResult:
    ensure_collection()
    desired_ids: Set[str] = set()

    def tracked_points() -> Iterator[models.PointStruct]:
        for point in _database_points(session):
            desired_ids.add(str(point.id))
            yield point

    indexed_points = _upsert(tracked_points())
    stale_ids = _stored_point_ids() - desired_ids
    if stale_ids:
        _client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=models.PointIdsList(points=sorted(stale_ids)),
            wait=True,
        )
    return VectorSyncResult(
        indexed_points=indexed_points,
        removed_points=len(stale_ids),
    )


def synchronize_all_safely(session: Session) -> bool:
    try:
        result = synchronize_all(session)
        logger.info(
            "Qdrant synchronization completed: indexed=%s removed=%s",
            result.indexed_points,
            result.removed_points,
        )
        return True
    except Exception:
        logger.exception("Qdrant synchronization failed; pgvector fallback remains active")
        return False


def _face_id_filter(face_id: UUID) -> models.Filter:
    return models.Filter(
        must=[
            models.FieldCondition(
                key="faceId",
                match=models.MatchValue(value=str(face_id)),
            )
        ]
    )


def synchronize_face_id(session: Session, face_id: UUID) -> None:
    ensure_collection()
    existing_ids = _stored_point_ids(_face_id_filter(face_id))

    points: List[models.PointStruct] = []
    person = session.scalar(select(Person).where(Person.face_id == face_id))
    if person is not None:
        points.extend(
            _face_image_point(face_image, person)
            for face_image in session.scalars(
                select(FaceImage).where(FaceImage.person_id == person.id)
            )
        )

    identity = session.scalar(
        select(AnonymousIdentity).where(AnonymousIdentity.face_id == face_id)
    )
    if identity is not None:
        points.extend(
            _anonymous_point(sample, identity)
            for sample in session.scalars(
                select(AnonymousFaceEmbedding).where(
                    AnonymousFaceEmbedding.anonymous_identity_id == identity.id
                )
            )
        )

    desired_ids = {str(point.id) for point in points}
    _upsert(points)
    stale_ids = existing_ids - desired_ids
    if stale_ids:
        _client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=models.PointIdsList(points=sorted(stale_ids)),
            wait=True,
        )


def synchronize_face_id_safely(session: Session, face_id: UUID) -> bool:
    try:
        synchronize_face_id(session, face_id)
        return True
    except Exception:
        logger.exception(
            "Qdrant face synchronization failed for %s; pgvector fallback remains active",
            face_id,
        )
        return False


def search_vectors(
    embedding: List[float],
    status: str,
    exclude_person_id: Optional[int] = None,
    limit: int = 10,
    owner_user_id: Optional[UUID] = None,
) -> List[VectorSearchResult]:
    must = [
        models.FieldCondition(
            key="status",
            match=models.MatchValue(value=status),
        )
    ]
    must_not = []
    should = []
    if owner_user_id is not None:
        if status == "known":
            should = [
                models.FieldCondition(
                    key="isGlobal", match=models.MatchValue(value=True)
                ),
                models.FieldCondition(
                    key="ownerId",
                    match=models.MatchValue(value=str(owner_user_id)),
                ),
            ]
        else:
            must.append(
                models.FieldCondition(
                    key="ownerId",
                    match=models.MatchValue(value=str(owner_user_id)),
                )
            )
    if exclude_person_id is not None:
        must_not.append(
            models.FieldCondition(
                key="personId",
                match=models.MatchValue(value=exclude_person_id),
            )
        )
    try:
        ensure_collection()
        results = _client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=embedding,
            query_filter=models.Filter(must=must, must_not=must_not, should=should),
            limit=limit,
            with_payload=True,
        )
    except Exception as error:
        raise VectorStoreUnavailable("Qdrant search failed") from error

    return [
        VectorSearchResult(
            point_id=UUID(str(result.id)),
            score=max(-1.0, min(1.0, float(result.score))),
            payload=dict(result.payload or {}),
        )
        for result in results
    ]


def qdrant_is_ready() -> bool:
    try:
        ensure_collection()
        _client.get_collection(QDRANT_COLLECTION)
        return True
    except Exception:
        return False


def qdrant_point_count() -> Optional[int]:
    try:
        collection = _client.get_collection(QDRANT_COLLECTION)
        return int(collection.points_count or 0)
    except Exception:
        return None
