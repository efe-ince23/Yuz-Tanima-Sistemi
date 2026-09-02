from typing import Optional, Tuple

import numpy as np
from fastapi import UploadFile

from app.api_errors import api_error
from app.face_detector import (
    AmbiguousFacesError,
    FaceCountError,
    PrimaryFaceData,
    decode_image,
    extract_primary_face_data,
    extract_single_face_data,
)


MAX_IMAGE_SIZE = 10 * 1024 * 1024
SUPPORTED_IMAGE_TYPES = {
    "image/jpeg": "jpeg",
    "image/png": "png",
    "image/webp": "webp",
}


def detect_image_format(content: bytes) -> Optional[str]:
    if content.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if len(content) >= 12 and content.startswith(b"RIFF") and content[8:12] == b"WEBP":
        return "webp"
    return None


async def read_uploaded_image(file: UploadFile, field_name: str) -> np.ndarray:
    content_type = (file.content_type or "").lower().split(";", 1)[0].strip()
    declared_format = SUPPORTED_IMAGE_TYPES.get(content_type)
    if declared_format is None:
        raise api_error(
            415,
            "UNSUPPORTED_IMAGE_TYPE",
            (
                f"{field_name}: Desteklenmeyen dosya turu. "
                "Yalnizca JPEG, PNG veya WebP kabul edilir."
            ),
            {"field": field_name, "supported_types": list(SUPPORTED_IMAGE_TYPES)},
        )

    content = await file.read()
    if not content:
        raise api_error(
            400,
            "EMPTY_FILE",
            f"{field_name}: Yuklenen dosya bos.",
            {"field": field_name},
        )

    if len(content) > MAX_IMAGE_SIZE:
        raise api_error(
            413,
            "FILE_TOO_LARGE",
            f"{field_name}: Resim en fazla 10 MB olabilir.",
            {"field": field_name, "max_bytes": MAX_IMAGE_SIZE},
        )

    detected_format = detect_image_format(content)
    if detected_format is None:
        raise api_error(
            400,
            "CORRUPT_IMAGE",
            f"{field_name}: Goruntu dosyasi bozuk veya gecersiz.",
            {"field": field_name},
        )
    if detected_format != declared_format:
        raise api_error(
            415,
            "IMAGE_CONTENT_TYPE_MISMATCH",
            f"{field_name}: Dosya icerigi bildirilen goruntu turuyle uyusmuyor.",
            {
                "field": field_name,
                "declared_format": declared_format,
                "detected_format": detected_format,
            },
        )

    image = decode_image(content)
    if image is None:
        raise api_error(
            400,
            "UNREADABLE_IMAGE",
            f"{field_name}: Resim dosyasi okunamadi.",
            {"field": field_name},
        )
    return image


def extract_face_data_or_error(
    image: np.ndarray, field_name: str
) -> Tuple[np.ndarray, float]:
    try:
        return extract_single_face_data(image)
    except FaceCountError as error:
        code = "FACE_NOT_FOUND" if error.face_count == 0 else "MULTIPLE_FACES"
        raise api_error(
            422,
            code,
            (
                f"{field_name}: Tam olarak bir yuz bulunmalidir. "
                f"Bulunan yuz sayisi: {error.face_count}."
            ),
            {"field": field_name, "face_count": error.face_count},
        ) from error


def extract_primary_face_data_or_error(
    image: np.ndarray, field_name: str
) -> PrimaryFaceData:
    try:
        return extract_primary_face_data(image)
    except AmbiguousFacesError as error:
        raise api_error(
            422,
            "AMBIGUOUS_FACES",
            (
                f"{field_name}: Birden fazla belirgin yuz bulundu. "
                "Ana yuz guvenle secilemedi."
            ),
            {"field": field_name},
        ) from error
    except FaceCountError as error:
        code = "FACE_NOT_FOUND" if error.face_count == 0 else "MULTIPLE_FACES"
        raise api_error(
            422,
            code,
            (
                f"{field_name}: Tam olarak bir yuz bulunmalidir. "
                f"Bulunan yuz sayisi: {error.face_count}."
            ),
            {"field": field_name, "face_count": error.face_count},
        ) from error


def extract_primary_face_data_or_none(
    image: np.ndarray, field_name: str
) -> Optional[PrimaryFaceData]:
    try:
        return extract_primary_face_data(image)
    except AmbiguousFacesError as error:
        raise api_error(
            422,
            "AMBIGUOUS_FACES",
            (
                f"{field_name}: Birden fazla belirgin yuz bulundu. "
                "Ana yuz guvenle secilemedi."
            ),
            {"field": field_name},
        ) from error
    except FaceCountError as error:
        if error.face_count == 0:
            return None
        raise api_error(
            422,
            "MULTIPLE_FACES",
            (
                f"{field_name}: Tam olarak bir yuz bulunmalidir. "
                f"Bulunan yuz sayisi: {error.face_count}."
            ),
            {"field": field_name, "face_count": error.face_count},
        ) from error
