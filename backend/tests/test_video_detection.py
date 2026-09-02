import unittest
from unittest.mock import patch

import numpy as np

from app.video_detection import detect_video_faces
from app.video_frames import SampledVideoFrame, VideoSamplingSummary
from app.yolo_arcface import DetectedYoloFace


class _FakeEngine:
    def __init__(self):
        self.recognizer = self
        self.batch_sizes = []

    def detect(self, image: np.ndarray):
        if int(image[0, 0, 0]) == 0:
            return []
        return [
            DetectedYoloFace(
                bbox=np.asarray([-10.0, 20.0, 220.0, 90.0], dtype=np.float32),
                landmarks=np.asarray(
                    [[10, 30], [180, 30], [100, 50], [30, 80], [170, 80]],
                    dtype=np.float32,
                ),
                confidence=0.91,
            ),
            DetectedYoloFace(
                bbox=np.asarray([20.0, 10.0, 70.0, 60.0], dtype=np.float32),
                landmarks=np.asarray(
                    [[30, 20], [60, 20], [45, 35], [35, 50], [55, 50]],
                    dtype=np.float32,
                ),
                confidence=0.87,
            ),
        ]

    def align(self, image, face):
        del image
        return np.full((112, 112, 3), int(face.bbox[0]), dtype=np.uint8)

    def embed_aligned(self, aligned_faces):
        self.batch_sizes.append(len(aligned_faces))
        return np.asarray(
            [
                np.full(512, index + 1, dtype=np.float32)
                for index, _ in enumerate(aligned_faces)
            ]
        )


def _fake_sample_video_object(object_path, sample_fps, frame_handler):
    del object_path, sample_fps
    frame_handler(
        SampledVideoFrame(
            frame_number=0,
            timestamp_ms=0,
            image=np.zeros((100, 200, 3), dtype=np.uint8),
        )
    )
    frame_handler(
        SampledVideoFrame(
            frame_number=10,
            timestamp_ms=1000,
            image=np.ones((100, 200, 3), dtype=np.uint8),
        )
    )
    return VideoSamplingSummary(
        source_fps=10.0,
        source_frame_count=20,
        decoded_frame_count=20,
        sampled_frame_count=2,
        width=200,
        height=100,
        last_timestamp_ms=1900,
    )


class VideoFaceDetectionTests(unittest.TestCase):
    def test_detects_every_face_and_keeps_frame_time(self) -> None:
        frames = []
        engine = _FakeEngine()
        with patch(
            "app.video_detection.sample_video_object",
            side_effect=_fake_sample_video_object,
        ), patch(
            "app.video_detection.get_yolo_arcface_engine",
            return_value=engine,
        ):
            summary = detect_video_faces(
                "videos/test/source.mp4",
                sample_fps=2.0,
                frame_handler=frames.append,
            )

        self.assertEqual(summary.sampled_frame_count, 2)
        self.assertEqual(summary.frames_with_faces, 1)
        self.assertEqual(summary.detected_face_count, 2)
        self.assertEqual(frames[0].faces, ())
        self.assertEqual(frames[1].frame_number, 10)
        self.assertEqual(frames[1].timestamp_ms, 1000)
        self.assertEqual(len(frames[1].faces), 2)
        self.assertEqual(frames[1].faces[0].bounding_box, (0, 20, 200, 90))
        self.assertEqual(
            frames[1].faces[0].normalized_bounding_box,
            (0.0, 0.2, 1.0, 0.9),
        )
        self.assertEqual(frames[1].faces[1].face_index, 1)
        self.assertEqual(len(frames[1].faces[0].landmarks), 5)
        self.assertEqual(frames[1].faces[0].embedding.shape, (512,))
        self.assertEqual(engine.batch_sizes, [2])


if __name__ == "__main__":
    unittest.main()
