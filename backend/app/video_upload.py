import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Optional

from fastapi import UploadFile

from app.video_config import VideoSettings


UPLOAD_CHUNK_SIZE = 1024 * 1024
PROBE_TIMEOUT_SECONDS = 30
TRANSCODE_TIMEOUT_SECONDS = 600
LIVE_RECORDING_OUTPUT_FPS = 30


class VideoUploadError(ValueError):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details


@dataclass(frozen=True)
class VideoMetadata:
    container: str
    codec: str
    duration_seconds: float
    source_fps: float
    width: int
    height: int
    frame_count: Optional[int]


@dataclass(frozen=True)
class ValidatedVideo:
    temporary_path: Path
    original_filename: str
    content_type: str
    file_size_bytes: int
    metadata: VideoMetadata

    def cleanup(self) -> None:
        self.temporary_path.unlink(missing_ok=True)


@dataclass(frozen=True)
class ReceivedLiveRecording:
    temporary_path: Path
    original_filename: str
    content_type: str
    file_size_bytes: int

    def cleanup(self) -> None:
        self.temporary_path.unlink(missing_ok=True)


def _parse_rate(value: object) -> float:
    text = str(value or "0")
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        denominator_value = float(denominator)
        return float(numerator) / denominator_value if denominator_value else 0.0
    return float(text)


def _normalized_container(probe: Dict[str, Any], suffix: str) -> str:
    format_name = str(probe.get("format", {}).get("format_name", "")).lower()
    if suffix == ".mp4" and "mp4" in format_name.split(","):
        return "mp4"
    return format_name.split(",", 1)[0]


def _probe_video(path: Path, settings: VideoSettings) -> VideoMetadata:
    try:
        probe_result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_SECONDS,
        )
        probe = json.loads(probe_result.stdout)
    except (
        FileNotFoundError,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as error:
        raise VideoUploadError(
            400,
            "VIDEO_UNREADABLE",
            "Video okunamadi veya bozuk.",
        ) from error

    video_stream = next(
        (
            stream
            for stream in probe.get("streams", [])
            if stream.get("codec_type") == "video"
        ),
        None,
    )
    if video_stream is None:
        raise VideoUploadError(400, "VIDEO_STREAM_MISSING", "Videoda goruntu akisi yok.")

    container = _normalized_container(probe, path.suffix.lower())
    codec = str(video_stream.get("codec_name", "")).lower()
    if not settings.supports_container(container):
        raise VideoUploadError(
            415,
            "UNSUPPORTED_VIDEO_CONTAINER",
            "Video kapsayici formati desteklenmiyor.",
            {"container": container, "allowed": list(settings.allowed_containers)},
        )
    if not settings.supports_codec(codec):
        raise VideoUploadError(
            415,
            "UNSUPPORTED_VIDEO_CODEC",
            "Video codec formati desteklenmiyor.",
            {"codec": codec, "allowed": list(settings.allowed_codecs)},
        )

    try:
        source_fps = _parse_rate(
            video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate")
        )
        duration_seconds = float(
            video_stream.get("duration")
            or probe.get("format", {}).get("duration")
            or 0
        )
        width = int(video_stream.get("width") or 0)
        height = int(video_stream.get("height") or 0)
        raw_frame_count = video_stream.get("nb_frames")
        frame_count = int(raw_frame_count) if raw_frame_count is not None else None
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise VideoUploadError(
            400,
            "VIDEO_METADATA_INVALID",
            "Video bilgileri okunamadi.",
        ) from error

    if duration_seconds <= 0 or source_fps <= 0 or width <= 0 or height <= 0:
        raise VideoUploadError(
            400,
            "VIDEO_METADATA_INVALID",
            "Video sure, kare hizi veya boyut bilgisi gecersiz.",
        )
    if duration_seconds > settings.max_duration_seconds:
        raise VideoUploadError(
            413,
            "VIDEO_DURATION_EXCEEDED",
            "Video izin verilen sureden uzun.",
            {
                "duration_seconds": round(duration_seconds, 3),
                "max_duration_seconds": settings.max_duration_seconds,
            },
        )

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(path),
                "-frames:v",
                "1",
                "-f",
                "null",
                "-",
            ],
            check=True,
            capture_output=True,
            timeout=PROBE_TIMEOUT_SECONDS,
        )
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as error:
        raise VideoUploadError(
            400,
            "VIDEO_DECODE_FAILED",
            "Videonun ilk karesi okunamadi.",
        ) from error

    return VideoMetadata(
        container=container,
        codec=codec,
        duration_seconds=duration_seconds,
        source_fps=source_fps,
        width=width,
        height=height,
        frame_count=frame_count,
    )


async def validate_uploaded_video(
    file: UploadFile,
    settings: VideoSettings,
) -> ValidatedVideo:
    original_filename = Path(file.filename or "video.mp4").name[:255]
    content_type = (file.content_type or "").lower().split(";", 1)[0].strip()
    if not settings.supports_content_type(content_type):
        raise VideoUploadError(
            415,
            "UNSUPPORTED_VIDEO_MEDIA_TYPE",
            "Yalnizca desteklenen video medya turleri kabul edilir.",
            {"content_type": content_type, "allowed": list(settings.allowed_content_types)},
        )
    if Path(original_filename).suffix.lower() != ".mp4":
        raise VideoUploadError(
            415,
            "UNSUPPORTED_VIDEO_EXTENSION",
            "Video dosyasi MP4 uzantili olmalidir.",
        )

    temporary_file = NamedTemporaryFile(prefix="face-video-", suffix=".mp4", delete=False)
    temporary_path = Path(temporary_file.name)
    file_size_bytes = 0
    try:
        with temporary_file:
            while True:
                chunk = await file.read(UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                file_size_bytes += len(chunk)
                if file_size_bytes > settings.max_size_bytes:
                    raise VideoUploadError(
                        413,
                        "VIDEO_FILE_TOO_LARGE",
                        "Video dosyasi izin verilen boyutu asiyor.",
                        {"max_size_bytes": settings.max_size_bytes},
                    )
                temporary_file.write(chunk)

        if file_size_bytes == 0:
            raise VideoUploadError(400, "VIDEO_EMPTY", "Video dosyasi bos.")
        with temporary_path.open("rb") as uploaded:
            header = uploaded.read(12)
        if len(header) < 12 or header[4:8] != b"ftyp":
            raise VideoUploadError(
                400,
                "VIDEO_SIGNATURE_INVALID",
                "Dosya gecerli bir MP4 videosu degil.",
            )

        metadata = _probe_video(temporary_path, settings)
        return ValidatedVideo(
            temporary_path=temporary_path,
            original_filename=original_filename,
            content_type=content_type,
            file_size_bytes=file_size_bytes,
            metadata=metadata,
        )
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


async def receive_live_recording(
    file: UploadFile,
    settings: VideoSettings,
) -> ReceivedLiveRecording:
    original_filename = Path(file.filename or "live-recording.webm").name[:255]
    content_type = (file.content_type or "").lower().split(";", 1)[0].strip()
    suffix = Path(original_filename).suffix.lower()
    if content_type != "video/webm" or suffix != ".webm":
        raise VideoUploadError(
            415,
            "UNSUPPORTED_LIVE_RECORDING_FORMAT",
            "Canli kamera kaydi WebM formatinda olmalidir.",
            {"content_type": content_type, "extension": suffix},
        )

    source_file = NamedTemporaryFile(
        prefix="face-live-source-",
        suffix=".webm",
        delete=False,
    )
    source_path = Path(source_file.name)
    source_size = 0
    try:
        with source_file:
            while True:
                chunk = await file.read(UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                source_size += len(chunk)
                if source_size > settings.max_size_bytes:
                    raise VideoUploadError(
                        413,
                        "VIDEO_FILE_TOO_LARGE",
                        "Canli kamera kaydi izin verilen boyutu asiyor.",
                        {"max_size_bytes": settings.max_size_bytes},
                    )
                source_file.write(chunk)

        if source_size == 0:
            raise VideoUploadError(400, "VIDEO_EMPTY", "Canli kamera kaydi bos.")
        with source_path.open("rb") as uploaded:
            if uploaded.read(4) != b"\x1a\x45\xdf\xa3":
                raise VideoUploadError(
                    400,
                    "VIDEO_SIGNATURE_INVALID",
                    "Dosya gecerli bir WebM kamera kaydi degil.",
                )
        return ReceivedLiveRecording(
            temporary_path=source_path,
            original_filename=original_filename,
            content_type="video/webm",
            file_size_bytes=source_size,
        )
    except Exception:
        source_path.unlink(missing_ok=True)
        raise


def normalize_live_recording(
    source_path: Path,
    original_filename: str,
    settings: VideoSettings,
) -> ValidatedVideo:
    output_file = NamedTemporaryFile(
        prefix="face-live-normalized-",
        suffix=".mp4",
        delete=False,
    )
    output_path = Path(output_file.name)
    output_file.close()
    try:
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-i",
                    str(source_path),
                    "-map",
                    "0:v:0",
                    "-an",
                    "-vf",
                    f"fps={LIVE_RECORDING_OUTPUT_FPS}",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "ultrafast",
                    "-crf",
                    "23",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(output_path),
                ],
                check=True,
                capture_output=True,
                timeout=TRANSCODE_TIMEOUT_SECONDS,
            )
        except (
            FileNotFoundError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ) as error:
            raise VideoUploadError(
                400,
                "LIVE_RECORDING_TRANSCODE_FAILED",
                "Canli kamera kaydi MP4 formatina donusturulemedi.",
            ) from error

        normalized_size = output_path.stat().st_size
        if normalized_size <= 0:
            raise VideoUploadError(
                400,
                "VIDEO_EMPTY",
                "Donusturulen canli kamera kaydi bos.",
            )
        if normalized_size > settings.max_size_bytes:
            raise VideoUploadError(
                413,
                "VIDEO_FILE_TOO_LARGE",
                "Donusturulen kamera kaydi izin verilen boyutu asiyor.",
                {"max_size_bytes": settings.max_size_bytes},
            )

        metadata = _probe_video(output_path, settings)
        return ValidatedVideo(
            temporary_path=output_path,
            original_filename=f"{Path(original_filename).stem}.mp4",
            content_type="video/mp4",
            file_size_bytes=normalized_size,
            metadata=metadata,
        )
    except Exception:
        output_path.unlink(missing_ok=True)
        raise


async def validate_live_recording(
    file: UploadFile,
    settings: VideoSettings,
) -> ValidatedVideo:
    original_filename = Path(file.filename or "live-recording.webm").name[:255]
    content_type = (file.content_type or "").lower().split(";", 1)[0].strip()
    suffix = Path(original_filename).suffix.lower()
    if content_type == "video/mp4" and suffix == ".mp4":
        return await validate_uploaded_video(file, settings)

    received = await receive_live_recording(file, settings)
    try:
        return normalize_live_recording(
            received.temporary_path,
            received.original_filename,
            settings,
        )
    finally:
        received.cleanup()
