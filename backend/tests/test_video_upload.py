import json
import subprocess
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from fastapi import UploadFile
from starlette.datastructures import Headers

from app.video_config import VideoSettings
from app.video_upload import (
    VideoMetadata,
    VideoUploadError,
    _parse_rate,
    _probe_video,
    validate_live_recording,
    validate_uploaded_video,
)


def _settings(max_size_bytes: int = 1024) -> VideoSettings:
    return VideoSettings(
        max_size_bytes=max_size_bytes,
        max_duration_seconds=300.0,
        sample_fps=3.0,
        processing_concurrency=1,
        allowed_content_types=("video/mp4",),
        allowed_containers=("mp4",),
        allowed_codecs=("h264",),
    )


def _upload(content: bytes, filename: str = "test.mp4") -> UploadFile:
    return UploadFile(
        file=BytesIO(content),
        filename=filename,
        headers=Headers({"content-type": "video/mp4"}),
    )


def _live_upload(content: bytes, filename: str = "live.webm") -> UploadFile:
    return UploadFile(
        file=BytesIO(content),
        filename=filename,
        headers=Headers({"content-type": "video/webm"}),
    )


class VideoUploadTests(unittest.IsolatedAsyncioTestCase):
    def test_parses_fractional_frame_rate(self) -> None:
        self.assertAlmostEqual(_parse_rate("30000/1001"), 29.97002997)

    async def test_streams_and_accepts_a_valid_mp4(self) -> None:
        metadata = VideoMetadata("mp4", "h264", 2.0, 25.0, 640, 360, 50)
        content = b"\x00\x00\x00\x18ftypisom" + b"video-content"
        with patch("app.video_upload._probe_video", return_value=metadata):
            validated = await validate_uploaded_video(_upload(content), _settings())
        try:
            self.assertEqual(validated.file_size_bytes, len(content))
            self.assertEqual(validated.original_filename, "test.mp4")
            self.assertEqual(validated.metadata.codec, "h264")
            self.assertTrue(validated.temporary_path.is_file())
        finally:
            validated.cleanup()
        self.assertFalse(validated.temporary_path.exists())

    async def test_rejects_empty_spoofed_and_oversized_files(self) -> None:
        cases = (
            (b"", _settings(), "VIDEO_EMPTY"),
            (b"not-an-mp4", _settings(), "VIDEO_SIGNATURE_INVALID"),
            (b"\x00\x00\x00\x18ftypisom", _settings(8), "VIDEO_FILE_TOO_LARGE"),
        )
        for content, settings, expected_code in cases:
            with self.subTest(expected_code=expected_code):
                with self.assertRaises(VideoUploadError) as caught:
                    await validate_uploaded_video(_upload(content), settings)
                self.assertEqual(caught.exception.code, expected_code)

    async def test_normalizes_a_live_webm_recording_to_mp4(self) -> None:
        metadata = VideoMetadata("mp4", "h264", 2.0, 25.0, 640, 360, 50)
        transcode_command = None

        def transcode(command, **_kwargs):
            nonlocal transcode_command
            transcode_command = command
            Path(command[-1]).write_bytes(b"\x00\x00\x00\x18ftypisomnormalized")
            return subprocess.CompletedProcess(command, 0, b"", b"")

        with patch("app.video_upload.subprocess.run", side_effect=transcode), patch(
            "app.video_upload._probe_video",
            return_value=metadata,
        ):
            validated = await validate_live_recording(
                _live_upload(b"\x1a\x45\xdf\xa3webm-content"),
                _settings(),
            )
        try:
            self.assertEqual(validated.original_filename, "live.mp4")
            self.assertEqual(validated.content_type, "video/mp4")
            self.assertEqual(validated.metadata.codec, "h264")
            self.assertTrue(validated.temporary_path.is_file())
            self.assertIn("fps=30", transcode_command)
        finally:
            validated.cleanup()

    def test_rejects_an_unsupported_codec(self) -> None:
        probe = {
            "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "2"},
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "mpeg4",
                    "avg_frame_rate": "25/1",
                    "width": 640,
                    "height": 360,
                    "nb_frames": "50",
                }
            ],
        }
        completed = subprocess.CompletedProcess(
            args=["ffprobe"], returncode=0, stdout=json.dumps(probe), stderr=""
        )
        with tempfile.NamedTemporaryFile(suffix=".mp4") as video_file:
            with patch("app.video_upload.subprocess.run", return_value=completed):
                with self.assertRaises(VideoUploadError) as caught:
                    _probe_video(Path(video_file.name), _settings())
        self.assertEqual(caught.exception.code, "UNSUPPORTED_VIDEO_CODEC")


if __name__ == "__main__":
    unittest.main()
