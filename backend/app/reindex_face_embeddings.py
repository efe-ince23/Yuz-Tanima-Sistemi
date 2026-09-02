import argparse
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple
from uuid import UUID

import numpy as np
from qdrant_client import QdrantClient, models
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.face_detector import decode_image
from app.face_storage import read_face_image
from app.models import AnonymousFaceEmbedding, AnonymousIdentity, FaceImage, Person
from app.vector_store import FACE_VECTOR_SIZE, QDRANT_URL, _point_id
from app.yolo_arcface import DetectedYoloFace, YoloArcFaceEngine


MODEL_VERSION = "arcface_r50_w600k_v1"


@dataclass
class SourceSample:
    sample_type: str
    sample_id: int
    face_id: UUID
    person_id: Optional[int]
    status: str
    image_path: str
    database_record: object


def _source_samples(session: Session, limit: Optional[int]) -> List[SourceSample]:
    samples = [
        SourceSample(
            sample_type="face_image",
            sample_id=face_image.id,
            face_id=person.face_id,
            person_id=person.id,
            status="known",
            image_path=face_image.image_path,
            database_record=face_image,
        )
        for face_image, person in session.execute(
            select(FaceImage, Person)
            .join(Person, Person.id == FaceImage.person_id)
            .order_by(FaceImage.id)
        )
    ]
    samples.extend(
        SourceSample(
            sample_type="anonymous_embedding",
            sample_id=sample.id,
            face_id=identity.face_id,
            person_id=identity.person_id,
            status="known" if identity.person_id is not None else "anonymous",
            image_path=sample.image_path or "",
            database_record=sample,
        )
        for sample, identity in session.execute(
            select(AnonymousFaceEmbedding, AnonymousIdentity)
            .join(
                AnonymousIdentity,
                AnonymousIdentity.id
                == AnonymousFaceEmbedding.anonymous_identity_id,
            )
            .order_by(AnonymousFaceEmbedding.id)
        )
    )
    return samples[:limit] if limit is not None else samples


def _largest_face(engine: YoloArcFaceEngine, image: np.ndarray):
    faces = engine.detect(image)
    if not faces:
        raise RuntimeError("YOLOv8-Face did not detect a face")
    return max(
        faces,
        key=lambda face: max(0.0, float(face.bbox[2] - face.bbox[0]))
        * max(0.0, float(face.bbox[3] - face.bbox[1])),
    )


def _payload(sample: SourceSample) -> dict:
    return {
        "faceId": str(sample.face_id),
        "personId": sample.person_id,
        "status": sample.status,
        "sampleType": sample.sample_type,
        "sampleId": sample.sample_id,
        "imagePath": sample.image_path,
        "modelVersion": MODEL_VERSION,
    }


def _create_collection(client: QdrantClient, collection_name: str) -> None:
    existing = {item.name for item in client.get_collections().collections}
    if collection_name in existing:
        raise RuntimeError(
            f"Collection already exists: {collection_name}. Use a new name."
        )
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=FACE_VECTOR_SIZE,
            distance=models.Distance.COSINE,
        ),
    )
    for field_name, field_schema in (
        ("faceId", models.PayloadSchemaType.KEYWORD),
        ("status", models.PayloadSchemaType.KEYWORD),
        ("personId", models.PayloadSchemaType.INTEGER),
        ("modelVersion", models.PayloadSchemaType.KEYWORD),
    ):
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=field_schema,
            wait=True,
        )


def reindex(
    collection_name: str,
    limit: Optional[int],
    apply_database: bool,
    upsert_batch_size: int,
) -> None:
    session = SessionLocal()
    client = QdrantClient(url=QDRANT_URL, timeout=30.0)
    started_at = time.perf_counter()
    try:
        samples = _source_samples(session, limit)
        if not samples:
            raise RuntimeError("No face samples were found")
        missing_paths = [sample for sample in samples if not sample.image_path]
        if missing_paths:
            raise RuntimeError(
                f"{len(missing_paths)} samples do not have a stored source image"
            )

        _create_collection(client, collection_name)
        engine = YoloArcFaceEngine()
        pending_points: List[Tuple[SourceSample, DetectedYoloFace, np.ndarray]] = []
        processed = 0

        for sample in samples:
            content, _ = read_face_image(sample.image_path)
            image = decode_image(content)
            if image is None:
                raise RuntimeError(f"Unreadable image: {sample.image_path}")
            face = _largest_face(engine, image)
            aligned_face = engine.recognizer.align(image, face)
            pending_points.append((sample, face, aligned_face))

            if len(pending_points) >= upsert_batch_size:
                embeddings = engine.recognizer.embed_aligned(
                    [item[2] for item in pending_points]
                )
                points = []
                for (pending_sample, pending_face, _), embedding in zip(
                    pending_points, embeddings
                ):
                    points.append(
                        models.PointStruct(
                            id=str(
                                _point_id(
                                    pending_sample.sample_type,
                                    pending_sample.sample_id,
                                )
                            ),
                            vector=embedding.tolist(),
                            payload=_payload(pending_sample),
                        )
                    )
                    if apply_database:
                        pending_sample.database_record.embedding = embedding.tolist()
                        pending_sample.database_record.detection_confidence = float(
                            pending_face.confidence
                        )
                client.upsert(
                    collection_name=collection_name,
                    points=points,
                    wait=True,
                )
                processed += len(pending_points)
                pending_points.clear()
                print(f"Indexed {processed}/{len(samples)}", flush=True)

        if pending_points:
            embeddings = engine.recognizer.embed_aligned(
                [item[2] for item in pending_points]
            )
            points = []
            for (pending_sample, pending_face, _), embedding in zip(
                pending_points, embeddings
            ):
                points.append(
                    models.PointStruct(
                        id=str(
                            _point_id(
                                pending_sample.sample_type,
                                pending_sample.sample_id,
                            )
                        ),
                        vector=embedding.tolist(),
                        payload=_payload(pending_sample),
                    )
                )
                if apply_database:
                    pending_sample.database_record.embedding = embedding.tolist()
                    pending_sample.database_record.detection_confidence = float(
                        pending_face.confidence
                    )
            client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True,
            )
            processed += len(pending_points)

        collection = client.get_collection(collection_name)
        if int(collection.points_count or 0) != len(samples):
            raise RuntimeError(
                "Qdrant point count does not match the processed sample count"
            )

        if apply_database:
            session.commit()
        else:
            session.rollback()

        elapsed = time.perf_counter() - started_at
        print(
            f"Completed: collection={collection_name} points={processed} "
            f"database_updated={apply_database} elapsed_seconds={elapsed:.1f}",
            flush=True,
        )
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a versioned ArcFace R50 Qdrant collection safely."
    )
    parser.add_argument("--collection", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--apply-database", action="store_true")
    parser.add_argument("--upsert-batch-size", type=int, default=128)
    arguments = parser.parse_args()
    reindex(
        collection_name=arguments.collection,
        limit=arguments.limit,
        apply_database=arguments.apply_database,
        upsert_batch_size=max(1, arguments.upsert_batch_size),
    )


if __name__ == "__main__":
    main()
