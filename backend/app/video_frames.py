from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Callable, Optional

import cv2
import numpy as np

from app.face_storage import download_object_to_file


class VideoSamplingError(RuntimeError):
    pass


@dataclass(frozen=True)
class SampledVideoFrame:
    frame_number: int
    timestamp_ms: int
    image: np.ndarray


@dataclass(frozen=True)
class VideoSamplingSummary:
    source_fps: float
    source_frame_count: Optional[int]
    decoded_frame_count: int
    sampled_frame_count: int
    width: int
    height: int
    last_timestamp_ms: int


FrameHandler = Callable[[SampledVideoFrame], None]


def sample_video_file(
    video_path: Path,
    sample_fps: float,
    frame_handler: Optional[FrameHandler] = None,
) -> VideoSamplingSummary:
    if sample_fps <= 0:
        raise ValueError("Ornekleme FPS degeri sifirdan buyuk olmalidir.")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise VideoSamplingError("Video kareleri okunmak icin acilamadi.")

    try:
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        raw_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        source_frame_count = raw_frame_count if raw_frame_count > 0 else None
        if source_fps <= 0 or width <= 0 or height <= 0:
            raise VideoSamplingError("Video kare bilgileri gecersiz.")

        effective_sample_fps = min(sample_fps, source_fps)
        sample_interval_seconds = 1.0 / effective_sample_fps
        next_sample_seconds = 0.0
        decoded_frame_count = 0
        sampled_frame_count = 0
        last_timestamp_ms = 0
        epsilon = 0.5 / source_fps

        while True:
            grabbed = capture.grab()
            if not grabbed:
                break

            frame_number = decoded_frame_count
            timestamp_seconds = frame_number / source_fps
            decoded_frame_count += 1
            last_timestamp_ms = round(timestamp_seconds * 1000)
            if timestamp_seconds + epsilon < next_sample_seconds:
                continue

            readable, image = capture.retrieve()
            if not readable or image is None:
                raise VideoSamplingError("Orneklenen video karesi okunamadi.")

            sampled_frame = SampledVideoFrame(
                frame_number=frame_number,
                timestamp_ms=last_timestamp_ms,
                image=image,
            )
            if frame_handler is not None:
                frame_handler(sampled_frame)
            sampled_frame_count += 1
            while next_sample_seconds <= timestamp_seconds + epsilon:
                next_sample_seconds += sample_interval_seconds

        if decoded_frame_count == 0:
            raise VideoSamplingError("Videodan okunabilir kare alinamadi.")

        return VideoSamplingSummary(
            source_fps=source_fps,
            source_frame_count=source_frame_count,
            decoded_frame_count=decoded_frame_count,
            sampled_frame_count=sampled_frame_count,
            width=width,
            height=height,
            last_timestamp_ms=last_timestamp_ms,
        )
    finally:
        capture.release()


def sample_video_object(
    object_path: str,
    sample_fps: float,
    frame_handler: Optional[FrameHandler] = None,
) -> VideoSamplingSummary:
    suffix = PurePosixPath(object_path).suffix.lower()
    if suffix != ".mp4":
        raise VideoSamplingError("Yalnizca MP4 video nesneleri islenebilir.")

    with TemporaryDirectory(prefix="face-video-sampling-") as temporary_directory:
        local_video = Path(temporary_directory) / f"source{suffix}"
        try:
            download_object_to_file(object_path, local_video)
        except (FileNotFoundError, OSError, ValueError) as error:
            raise VideoSamplingError("Video nesne deposundan indirilemedi.") from error
        return sample_video_file(local_video, sample_fps, frame_handler)
