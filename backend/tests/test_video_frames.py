import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from app.video_frames import VideoSamplingError, sample_video_file


class VideoFrameSamplingTests(unittest.TestCase):
    def test_decodes_only_sampled_frames(self) -> None:
        class FakeCapture:
            def __init__(self):
                self.frame_number = 0
                self.retrieve_count = 0

            def isOpened(self):
                return True

            def get(self, property_id):
                return {
                    cv2.CAP_PROP_FPS: 10.0,
                    cv2.CAP_PROP_FRAME_WIDTH: 64,
                    cv2.CAP_PROP_FRAME_HEIGHT: 48,
                    cv2.CAP_PROP_FRAME_COUNT: 20,
                }.get(property_id, 0)

            def grab(self):
                if self.frame_number >= 20:
                    return False
                self.frame_number += 1
                return True

            def retrieve(self):
                self.retrieve_count += 1
                value = self.frame_number - 1
                return True, np.full((48, 64, 3), value, dtype=np.uint8)

            def release(self):
                return None

        capture = FakeCapture()
        with patch("app.video_frames.cv2.VideoCapture", return_value=capture):
            summary = sample_video_file(Path("sampling-test.mp4"), sample_fps=2.0)

        self.assertEqual(summary.decoded_frame_count, 20)
        self.assertEqual(summary.sampled_frame_count, 4)
        self.assertEqual(capture.retrieve_count, 4)

    def test_samples_frames_at_the_configured_rate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            video_path = Path(directory) / "sampling-test.avi"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                10.0,
                (64, 48),
            )
            self.assertTrue(writer.isOpened())
            for frame_number in range(20):
                image = np.full((48, 64, 3), frame_number, dtype=np.uint8)
                writer.write(image)
            writer.release()

            sampled = []
            summary = sample_video_file(
                video_path,
                sample_fps=2.0,
                frame_handler=lambda frame: sampled.append(
                    (frame.frame_number, frame.timestamp_ms, frame.image.shape)
                ),
            )

        self.assertEqual(summary.decoded_frame_count, 20)
        self.assertEqual(summary.sampled_frame_count, 4)
        self.assertEqual([frame[0] for frame in sampled], [0, 5, 10, 15])
        self.assertEqual([frame[1] for frame in sampled], [0, 500, 1000, 1500])
        self.assertTrue(all(frame[2] == (48, 64, 3) for frame in sampled))

    def test_rejects_an_unreadable_video(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            invalid_path = Path(directory) / "invalid.mp4"
            invalid_path.write_bytes(b"not-a-video")
            with self.assertRaises(VideoSamplingError):
                sample_video_file(invalid_path, sample_fps=3.0)

    def test_rejects_an_invalid_sample_rate(self) -> None:
        with self.assertRaises(ValueError):
            sample_video_file(Path("unused.mp4"), sample_fps=0)


if __name__ == "__main__":
    unittest.main()
