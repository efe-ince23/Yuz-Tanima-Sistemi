import argparse
import csv
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError

from app.database import SessionLocal, engine
from app.face_detector import decode_image, extract_all_faces_data
from app.face_storage import object_exists, save_object
from app.models import DatasetImportItem, FaceImage, Person
from app.vector_store import synchronize_face_id_safely


DATASET_NAME = "lfw"
DEFAULT_ROOT = "/datasets/lfw/lfw-deepfunneled/lfw-deepfunneled"
TERMINAL_STATUSES = {"imported", "no_face", "corrupt", "error"}
logger = logging.getLogger(__name__)


@dataclass
class ImportSummary:
    selected_people: int = 0
    created_people: int = 0
    processed_images: int = 0
    imported_images: int = 0
    skipped_images: int = 0
    no_face_images: int = 0
    corrupt_images: int = 0
    failed_images: int = 0
    qdrant_sync_failures: int = 0


def _person_name(label: str) -> Tuple[str, str]:
    parts = label.replace("_", " ").split()
    if not parts:
        raise ValueError("LFW kimlik etiketi bos olamaz.")
    return parts[0], " ".join(parts[1:])


def _image_files(identity_directory: Path) -> List[Path]:
    return sorted(
        path
        for path in identity_directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
    )


def _processed_image_counts() -> Dict[str, int]:
    with SessionLocal() as session:
        rows = session.execute(
            select(
                DatasetImportItem.external_identity,
                func.count(DatasetImportItem.id),
            ).where(
                DatasetImportItem.dataset_name == DATASET_NAME,
                DatasetImportItem.status.in_(TERMINAL_STATUSES),
            ).group_by(DatasetImportItem.external_identity)
        ).all()
        return {identity: int(count) for identity, count in rows}


def _candidate_directories(root: Path, max_people: Optional[int]) -> List[Path]:
    candidates: List[Path] = []
    processed_counts = _processed_image_counts()
    metadata_path = root.parent.parent / "lfw_allnames.csv"
    expected_counts: Dict[str, int] = {}
    if metadata_path.is_file():
        with metadata_path.open("r", encoding="utf-8-sig", newline="") as metadata_file:
            for row in csv.DictReader(metadata_file):
                expected_counts[row["name"]] = int(row["images"])
    else:
        expected_counts = {
            directory.name: len(_image_files(directory))
            for directory in root.iterdir()
            if directory.is_dir()
        }

    for label, expected_images in sorted(expected_counts.items()):
        if processed_counts.get(label, 0) >= expected_images:
            continue
        directory = root / label
        if not directory.is_dir():
            logger.error("LFW kimlik klasoru bulunamadi: %s", directory)
            continue
        candidates.append(directory)
        if max_people is not None and len(candidates) >= max_people:
            break
    return candidates


def _get_or_create_person(label: str) -> Tuple[int, bool]:
    with SessionLocal() as session:
        person = session.scalar(
            select(Person).where(
                Person.source == DATASET_NAME,
                Person.external_id == label,
            )
        )
        if person is not None:
            return person.id, False

        first_name, last_name = _person_name(label)
        person = Person(
            first_name=first_name,
            last_name=last_name,
            description="LFW Dataset",
            source=DATASET_NAME,
            external_id=label,
        )
        session.add(person)
        try:
            session.commit()
        except IntegrityError:
            session.rollback()
            existing = session.scalar(
                select(Person).where(
                    Person.source == DATASET_NAME,
                    Person.external_id == label,
                )
            )
            if existing is None:
                raise
            return existing.id, False
        session.refresh(person)
        return person.id, True


def _largest_face(image_content: bytes):
    image = decode_image(image_content)
    if image is None or image.size == 0:
        return "corrupt", None
    faces = [face for face in extract_all_faces_data(image) if face.embedding is not None]
    if not faces:
        return "no_face", None
    return "imported", max(
        faces,
        key=lambda face: max(0, face.bounding_box[2] - face.bounding_box[0])
        * max(0, face.bounding_box[3] - face.bounding_box[1]),
    )


def _import_image(person_id: int, label: str, image_path: Path) -> str:
    source_path = f"{label}/{image_path.name}"
    object_path = f"persons/{person_id}/lfw/{image_path.name}"

    with SessionLocal() as session:
        item = session.scalar(
            select(DatasetImportItem).where(
                DatasetImportItem.dataset_name == DATASET_NAME,
                DatasetImportItem.source_path == source_path,
            )
        )
        if item is not None and item.status in TERMINAL_STATUSES:
            return "skipped"
        if item is None:
            item = DatasetImportItem(
                dataset_name=DATASET_NAME,
                external_identity=label,
                source_path=source_path,
                person_id=person_id,
                status="processing",
            )
            session.add(item)
        else:
            item.status = "processing"
            item.error_detail = None
        session.commit()

    try:
        content = image_path.read_bytes()
        if not content:
            final_status, detected_face = "corrupt", None
        else:
            if not object_exists(object_path):
                save_object(object_path, content, "image/jpeg")
            final_status, detected_face = _largest_face(content)

        with SessionLocal() as session:
            item = session.scalar(
                select(DatasetImportItem).where(
                    DatasetImportItem.dataset_name == DATASET_NAME,
                    DatasetImportItem.source_path == source_path,
                )
            )
            if item is None:
                raise RuntimeError("Aktarim takip kaydi bulunamadi.")
            item.image_path = object_path if content else None
            item.status = final_status
            if detected_face is not None:
                face_image = FaceImage(
                    person_id=person_id,
                    image_path=object_path,
                    embedding=detected_face.embedding.tolist(),
                    detection_confidence=detected_face.confidence,
                )
                session.add(face_image)
                session.flush()
                item.face_image_id = face_image.id
            session.commit()
        return final_status
    except Exception as error:
        logger.exception("LFW resmi aktarilamadi: %s", source_path)
        with SessionLocal() as session:
            item = session.scalar(
                select(DatasetImportItem).where(
                    DatasetImportItem.dataset_name == DATASET_NAME,
                    DatasetImportItem.source_path == source_path,
                )
            )
            if item is not None:
                item.status = "error"
                item.error_detail = str(error)[:2000]
                item.image_path = object_path if object_exists(object_path) else None
                session.commit()
        return "error"


def import_lfw(root: Path, max_people: Optional[int]) -> ImportSummary:
    if not root.is_dir():
        raise FileNotFoundError(f"LFW klasoru bulunamadi: {root}")

    with engine.connect() as lock_connection:
        acquired = lock_connection.scalar(
            text("SELECT pg_try_advisory_lock(hashtext(:lock_name))"),
            {"lock_name": "lfw_dataset_import"},
        )
        if not acquired:
            raise RuntimeError("Baska bir LFW aktarimi zaten calisiyor.")
        try:
            candidates = _candidate_directories(root, max_people)
            summary = ImportSummary(selected_people=len(candidates))
            for person_index, directory in enumerate(candidates, start=1):
                person_id, created = _get_or_create_person(directory.name)
                summary.created_people += int(created)
                images = _image_files(directory)
                for image_path in images:
                    status = _import_image(person_id, directory.name, image_path)
                    summary.processed_images += int(status != "skipped")
                    summary.imported_images += int(status == "imported")
                    summary.skipped_images += int(status == "skipped")
                    summary.no_face_images += int(status == "no_face")
                    summary.corrupt_images += int(status == "corrupt")
                    summary.failed_images += int(status == "error")

                with SessionLocal() as session:
                    face_id = session.scalar(
                        select(Person.face_id).where(Person.id == person_id)
                    )
                    if face_id is None or not synchronize_face_id_safely(session, face_id):
                        summary.qdrant_sync_failures += 1
                print(
                    json.dumps(
                        {
                            "progress": f"{person_index}/{len(candidates)}",
                            "identity": directory.name,
                            "images": len(images),
                        },
                        ensure_ascii=True,
                    ),
                    flush=True,
                )
            return summary
        finally:
            lock_connection.execute(
                text("SELECT pg_advisory_unlock(hashtext(:lock_name))"),
                {"lock_name": "lfw_dataset_import"},
            )


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="LFW verisini guvenli ve devam edilebilir aktarir.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.getenv("LFW_DATASET_ROOT", DEFAULT_ROOT)),
    )
    parser.add_argument("--max-people", type=int, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.max_people is not None and args.max_people <= 0:
        parser.error("--max-people pozitif olmalidir")

    logging.basicConfig(level=logging.INFO)
    summary = import_lfw(args.root, args.max_people)
    print(json.dumps(asdict(summary), ensure_ascii=True), flush=True)
    return 0 if summary.failed_images == 0 and summary.qdrant_sync_failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
