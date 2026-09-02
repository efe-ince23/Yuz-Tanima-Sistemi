import mimetypes
import os
import shutil
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Iterable, List, Optional, Tuple
from threading import Lock
from uuid import uuid4

import cv2
import numpy as np
from minio import Minio
from minio.commonconfig import CopySource
from minio.error import S3Error


DATA_ROOT = Path(os.getenv("DATA_ROOT", "/app/data"))
FACE_IMAGES_ROOT = DATA_ROOT / "persons"
ANONYMOUS_IMAGES_ROOT = DATA_ROOT / "anonymous"
TRASH_ROOT = DATA_ROOT / ".trash"
STORAGE_BACKEND = os.getenv("OBJECT_STORAGE_BACKEND", "local").strip().lower()
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "face-images")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").strip().lower() in {
    "1",
    "true",
    "yes",
}

if STORAGE_BACKEND not in {"local", "minio"}:
    raise RuntimeError("OBJECT_STORAGE_BACKEND local veya minio olmalidir.")

_minio_client: Optional[Minio] = None
_object_count_lock = Lock()
_object_count_cache: Tuple[float, Optional[int]] = (0.0, None)
OBJECT_COUNT_CACHE_SECONDS = float(os.getenv("OBJECT_COUNT_CACHE_SECONDS", "30"))


@dataclass(frozen=True)
class StagedFile:
    original_path: str
    staged_path: str


@dataclass(frozen=True)
class MigrationResult:
    migrated: int
    already_present: int
    missing_local: int


def storage_backend_name() -> str:
    return STORAGE_BACKEND


def _client() -> Minio:
    global _minio_client
    if _minio_client is None:
        if not MINIO_ACCESS_KEY or not MINIO_SECRET_KEY:
            raise RuntimeError("MinIO erisim bilgileri eksik.")
        _minio_client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=MINIO_SECURE,
        )
    return _minio_client


def _normalize_key(relative_path: str) -> str:
    if not relative_path or "\\" in relative_path:
        raise ValueError("Gecersiz resim yolu.")
    path = PurePosixPath(relative_path)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError("Gecersiz resim yolu.")
    normalized = path.as_posix()
    if normalized in {"", "."}:
        raise ValueError("Gecersiz resim yolu.")
    return normalized


def ensure_data_directory() -> None:
    FACE_IMAGES_ROOT.mkdir(parents=True, exist_ok=True)
    ANONYMOUS_IMAGES_ROOT.mkdir(parents=True, exist_ok=True)


def ensure_object_storage() -> None:
    ensure_data_directory()
    if STORAGE_BACKEND != "minio":
        return
    client = _client()
    if not client.bucket_exists(MINIO_BUCKET):
        client.make_bucket(MINIO_BUCKET)


def storage_is_ready() -> bool:
    try:
        if STORAGE_BACKEND == "local":
            ensure_data_directory()
            return DATA_ROOT.exists()
        return _client().bucket_exists(MINIO_BUCKET)
    except Exception:
        return False


def storage_object_count() -> Optional[int]:
    global _object_count_cache
    checked_at, cached_count = _object_count_cache
    if time.monotonic() - checked_at < OBJECT_COUNT_CACHE_SECONDS:
        return cached_count
    try:
        if STORAGE_BACKEND == "local":
            count = sum(1 for path in DATA_ROOT.rglob("*") if path.is_file())
        else:
            count = sum(1 for _ in _client().list_objects(MINIO_BUCKET, recursive=True))
        with _object_count_lock:
            _object_count_cache = (time.monotonic(), count)
        return count
    except Exception:
        return None


def _invalidate_object_count() -> None:
    global _object_count_cache
    with _object_count_lock:
        _object_count_cache = (0.0, None)


def save_face_image(person_id: int, image: np.ndarray) -> str:
    return _save_jpeg(f"persons/{person_id}", image)


def save_object(relative_path: str, content: bytes, content_type: str) -> str:
    key = _normalize_key(relative_path)
    if not content:
        raise ValueError("Bos dosya saklanamaz.")
    if STORAGE_BACKEND == "minio":
        ensure_object_storage()
        _client().put_object(
            MINIO_BUCKET,
            key,
            BytesIO(content),
            length=len(content),
            content_type=content_type,
        )
        _invalidate_object_count()
        return key

    final_path = resolve_data_path(key)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = final_path.parent / f".{final_path.name}.tmp"
    try:
        temporary_path.write_bytes(content)
        temporary_path.replace(final_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    _invalidate_object_count()
    return key


def save_file_object(relative_path: str, source_path: Path, content_type: str) -> str:
    key = _normalize_key(relative_path)
    if not source_path.is_file() or source_path.stat().st_size <= 0:
        raise ValueError("Bos dosya saklanamaz.")
    if STORAGE_BACKEND == "minio":
        ensure_object_storage()
        _client().fput_object(
            MINIO_BUCKET,
            key,
            str(source_path),
            content_type=content_type,
        )
        _invalidate_object_count()
        return key

    final_path = resolve_data_path(key)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = final_path.parent / f".{final_path.name}.tmp"
    try:
        shutil.copyfile(source_path, temporary_path)
        temporary_path.replace(final_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    _invalidate_object_count()
    return key


def download_object_to_file(relative_path: str, destination: Path) -> Path:
    key = _normalize_key(relative_path)
    if destination.exists():
        raise ValueError("Hedef dosya zaten mevcut.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        if STORAGE_BACKEND == "minio":
            _client().fget_object(MINIO_BUCKET, key, str(destination))
        else:
            source = resolve_data_path(key)
            if not source.is_file():
                raise FileNotFoundError(key)
            shutil.copyfile(source, destination)
        if not destination.is_file() or destination.stat().st_size <= 0:
            raise OSError("Nesne bos veya indirilemedi.")
        return destination
    except S3Error as error:
        destination.unlink(missing_ok=True)
        if error.code in {"NoSuchKey", "NoSuchObject", "NoSuchBucket"}:
            raise FileNotFoundError(key) from error
        raise OSError("MinIO nesnesi indirilemedi.") from error
    except Exception:
        destination.unlink(missing_ok=True)
        raise


def object_exists(relative_path: str) -> bool:
    key = _normalize_key(relative_path)
    if STORAGE_BACKEND == "minio":
        return _minio_object_exists(key)
    return resolve_data_path(key).is_file()


def object_size(relative_path: str) -> int:
    key = _normalize_key(relative_path)
    if STORAGE_BACKEND == "local":
        path = resolve_data_path(key)
        if not path.is_file():
            raise FileNotFoundError(key)
        return path.stat().st_size

    try:
        return _client().stat_object(MINIO_BUCKET, key).size
    except S3Error as error:
        if error.code in {"NoSuchKey", "NoSuchObject", "NoSuchBucket"}:
            raise FileNotFoundError(key) from error
        raise OSError("MinIO nesne bilgisi okunamadi.") from error


def stream_object(
    relative_path: str,
    offset: int = 0,
    length: Optional[int] = None,
    chunk_size: int = 1024 * 1024,
):
    key = _normalize_key(relative_path)
    if offset < 0 or (length is not None and length < 0) or chunk_size <= 0:
        raise ValueError("Gecersiz nesne okuma araligi.")

    if STORAGE_BACKEND == "local":
        path = resolve_data_path(key)
        if not path.is_file():
            raise FileNotFoundError(key)
        with path.open("rb") as source:
            source.seek(offset)
            remaining = length
            while remaining is None or remaining > 0:
                requested = chunk_size if remaining is None else min(chunk_size, remaining)
                chunk = source.read(requested)
                if not chunk:
                    break
                yield chunk
                if remaining is not None:
                    remaining -= len(chunk)
        return

    response = None
    try:
        response = _client().get_object(
            MINIO_BUCKET,
            key,
            offset=offset,
            length=length,
        )
        remaining = length
        while remaining is None or remaining > 0:
            requested = chunk_size if remaining is None else min(chunk_size, remaining)
            chunk = response.read(requested)
            if not chunk:
                break
            yield chunk
            if remaining is not None:
                remaining -= len(chunk)
    except S3Error as error:
        if error.code in {"NoSuchKey", "NoSuchObject", "NoSuchBucket"}:
            raise FileNotFoundError(key) from error
        raise OSError("MinIO nesnesi okunamadi.") from error
    finally:
        if response is not None:
            response.close()
            response.release_conn()


def save_anonymous_face(
    face_id: object,
    image: np.ndarray,
    bounding_box: Tuple[int, int, int, int],
) -> str:
    image_height, image_width = image.shape[:2]
    x1, y1, x2, y2 = bounding_box
    padding = max(8, int(max(x2 - x1, y2 - y1) * 0.15))
    crop_x1 = max(0, x1 - padding)
    crop_y1 = max(0, y1 - padding)
    crop_x2 = min(image_width, x2 + padding)
    crop_y2 = min(image_height, y2 + padding)
    cropped_face = image[crop_y1:crop_y2, crop_x1:crop_x2]
    if cropped_face.size == 0:
        raise ValueError("Anonim yuz goruntusu kirpilamadi.")

    return save_anonymous_face_crop(face_id, cropped_face)


def save_anonymous_face_crop(face_id: object, cropped_face: np.ndarray) -> str:
    if cropped_face.size == 0:
        raise ValueError("Bos anonim yuz goruntusu saklanamaz.")
    return _save_jpeg(f"anonymous/{face_id}", cropped_face)


def save_recognition_photo(owner_user_id: object, process_id: object, image: np.ndarray) -> Tuple[str, int]:
    if image.size == 0:
        raise ValueError("Bos tanima fotografi saklanamaz.")
    encoded, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not encoded:
        raise RuntimeError("Tanima fotografi JPEG formatina donusturulemedi.")
    content = jpeg.tobytes()
    path = save_object(
        f"recognition-photos/{owner_user_id}/{process_id}/source.jpg",
        content,
        "image/jpeg",
    )
    return path, len(content)


def _save_jpeg(directory: str, image: np.ndarray) -> str:
    encoded, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not encoded:
        raise RuntimeError("Resim JPEG formatina donusturulemedi.")

    key = _normalize_key(f"{directory}/{uuid4().hex}.jpg")
    content = jpeg.tobytes()
    if STORAGE_BACKEND == "minio":
        ensure_object_storage()
        _client().put_object(
            MINIO_BUCKET,
            key,
            BytesIO(content),
            length=len(content),
            content_type="image/jpeg",
        )
        _invalidate_object_count()
        return key

    final_path = resolve_data_path(key)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = final_path.parent / f".{final_path.name}.tmp"
    try:
        temporary_path.write_bytes(content)
        temporary_path.replace(final_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    _invalidate_object_count()
    return key


def read_face_image(relative_path: str) -> Tuple[bytes, str]:
    key = _normalize_key(relative_path)
    content_type = mimetypes.guess_type(key)[0] or "application/octet-stream"
    if STORAGE_BACKEND == "local":
        path = resolve_data_path(key)
        if not path.is_file():
            raise FileNotFoundError(key)
        return path.read_bytes(), content_type

    response = None
    try:
        response = _client().get_object(MINIO_BUCKET, key)
        return response.read(), response.headers.get("content-type", content_type)
    except S3Error as error:
        if error.code in {"NoSuchKey", "NoSuchObject", "NoSuchBucket"}:
            raise FileNotFoundError(key) from error
        raise OSError("MinIO resmi okunamadi.") from error
    finally:
        if response is not None:
            response.close()
            response.release_conn()


def delete_face_image(relative_path: str) -> None:
    key = _normalize_key(relative_path)
    if STORAGE_BACKEND == "minio":
        try:
            _client().remove_object(MINIO_BUCKET, key)
            _invalidate_object_count()
        except S3Error as error:
            raise OSError("MinIO resmi silinemedi.") from error
        return
    resolve_data_path(key).unlink(missing_ok=True)
    _invalidate_object_count()


def resolve_data_path(relative_path: str) -> Path:
    key = _normalize_key(relative_path)
    data_root = DATA_ROOT.resolve()
    candidate = (data_root / key).resolve()
    if os.path.commonpath([str(data_root), str(candidate)]) != str(data_root):
        raise ValueError("Gecersiz resim yolu.")
    return candidate


def _minio_object_exists(key: str) -> bool:
    try:
        _client().stat_object(MINIO_BUCKET, key)
        return True
    except S3Error as error:
        if error.code in {"NoSuchKey", "NoSuchObject", "NoSuchBucket"}:
            return False
        raise


def migrate_legacy_objects(relative_paths: Iterable[str]) -> MigrationResult:
    if STORAGE_BACKEND != "minio":
        return MigrationResult(migrated=0, already_present=0, missing_local=0)

    ensure_object_storage()
    migrated = 0
    already_present = 0
    missing_local = 0
    for relative_path in dict.fromkeys(relative_paths):
        key = _normalize_key(relative_path)
        local_path = resolve_data_path(key)
        if _minio_object_exists(key):
            already_present += 1
            continue
        if not local_path.is_file():
            missing_local += 1
            continue
        content_type = mimetypes.guess_type(key)[0] or "application/octet-stream"
        _client().fput_object(
            MINIO_BUCKET,
            key,
            str(local_path),
            content_type=content_type,
        )
        migrated += 1
    if migrated:
        _invalidate_object_count()
    return MigrationResult(
        migrated=migrated,
        already_present=already_present,
        missing_local=missing_local,
    )


def stage_face_images_for_deletion(relative_paths: Iterable[str]) -> List[StagedFile]:
    staged_files: List[StagedFile] = []
    try:
        for relative_path in dict.fromkeys(relative_paths):
            key = _normalize_key(relative_path)
            staged_key = f".trash/{uuid4().hex}{PurePosixPath(key).suffix}"
            if STORAGE_BACKEND == "minio":
                if not _minio_object_exists(key):
                    continue
                _client().copy_object(
                    MINIO_BUCKET,
                    staged_key,
                    CopySource(MINIO_BUCKET, key),
                )
                staged_files.append(
                    StagedFile(original_path=key, staged_path=staged_key)
                )
                _client().remove_object(MINIO_BUCKET, key)
                continue
            else:
                original = resolve_data_path(key)
                if not original.exists():
                    continue
                staged = resolve_data_path(staged_key)
                staged.parent.mkdir(parents=True, exist_ok=True)
                original.replace(staged)
            staged_files.append(StagedFile(original_path=key, staged_path=staged_key))
    except (OSError, S3Error, ValueError) as error:
        restore_staged_files(staged_files)
        raise OSError("Resimler silmeye hazirlanamadi.") from error
    if staged_files:
        _invalidate_object_count()
    return staged_files


def restore_staged_files(staged_files: Iterable[StagedFile]) -> None:
    staged_file_list = list(staged_files)
    for staged_file in reversed(staged_file_list):
        if STORAGE_BACKEND == "minio":
            if not _minio_object_exists(staged_file.staged_path):
                continue
            _client().copy_object(
                MINIO_BUCKET,
                staged_file.original_path,
                CopySource(MINIO_BUCKET, staged_file.staged_path),
            )
            _client().remove_object(MINIO_BUCKET, staged_file.staged_path)
            continue

        staged = resolve_data_path(staged_file.staged_path)
        if not staged.exists():
            continue
        original = resolve_data_path(staged_file.original_path)
        original.parent.mkdir(parents=True, exist_ok=True)
        staged.replace(original)
    if staged_file_list:
        _invalidate_object_count()


def finalize_staged_files(staged_files: Iterable[StagedFile]) -> None:
    staged_file_list = list(staged_files)
    parent_directories = set()
    for staged_file in staged_file_list:
        if STORAGE_BACKEND == "minio":
            try:
                _client().remove_object(MINIO_BUCKET, staged_file.staged_path)
            except S3Error as error:
                raise OSError("MinIO gecici resmi temizlenemedi.") from error
            continue
        staged = resolve_data_path(staged_file.staged_path)
        staged.unlink(missing_ok=True)
        parent_directories.add(resolve_data_path(staged_file.original_path).parent)

    for directory in parent_directories:
        if directory == FACE_IMAGES_ROOT or not directory.exists():
            continue
        try:
            directory.rmdir()
        except OSError:
            pass

    if STORAGE_BACKEND == "local":
        try:
            TRASH_ROOT.rmdir()
        except OSError:
            pass
    if staged_file_list:
        _invalidate_object_count()


def image_url(relative_path: str) -> str:
    return f"/media/{_normalize_key(relative_path)}"
