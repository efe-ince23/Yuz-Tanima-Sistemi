import unittest
from unittest.mock import patch

import numpy as np

from app.video_config import VideoTrackingSettings
from app.video_detection import (
    DetectedVideoFrame,
    VideoDetectionSummary,
    VideoFaceDetection,
)
from app.video_tracking import (
    ByteTrackFaceTracker,
    MultiFaceTracker,
    VideoTrackingError,
    track_video_faces,
)


SETTINGS = VideoTrackingSettings(
    max_gap_ms=1500,
    min_iou=0.10,
    max_center_distance=0.25,
    max_area_ratio=3.0,
)


def _face(face_index, x1, y1, x2, y2, confidence=0.9, embedding=None):
    normalized_box = (x1, y1, x2, y2)
    return VideoFaceDetection(
        face_index=face_index,
        confidence=confidence,
        bounding_box=(
            round(x1 * 1000),
            round(y1 * 1000),
            round(x2 * 1000),
            round(y2 * 1000),
        ),
        normalized_bounding_box=normalized_box,
        landmarks=(),
        embedding=embedding,
    )


def _frame(frame_number, timestamp_ms, *faces):
    return DetectedVideoFrame(
        frame_number=frame_number,
        timestamp_ms=timestamp_ms,
        width=1000,
        height=1000,
        image=np.zeros((2, 2, 3), dtype=np.uint8),
        faces=tuple(faces),
    )


def _scene_frame(frame_number, timestamp_ms, color, *faces):
    return DetectedVideoFrame(
        frame_number=frame_number,
        timestamp_ms=timestamp_ms,
        width=1000,
        height=1000,
        image=np.full((64, 64, 3), color, dtype=np.uint8),
        faces=tuple(faces),
    )


class MultiFaceTrackerTests(unittest.TestCase):
    def test_bytetrack_keeps_the_same_face_id_across_frames(self):
        settings = VideoTrackingSettings(
            max_gap_ms=1500,
            min_iou=0.10,
            max_center_distance=0.25,
            max_area_ratio=3.0,
            engine="bytetrack",
        )
        tracker = ByteTrackFaceTracker(settings, sample_fps=6.0)

        first = tracker.update(
            _frame(0, 0, _face(0, 0.10, 0.20, 0.30, 0.50))
        )
        second = tracker.update(
            _frame(10, 167, _face(0, 0.12, 0.20, 0.32, 0.50))
        )

        self.assertEqual(len(first.faces), 1)
        self.assertEqual(first.faces[0].track_id, second.faces[0].track_id)
        self.assertTrue(first.faces[0].is_new_track)
        self.assertFalse(second.faces[0].is_new_track)
        self.assertEqual(tracker.summaries()[0].observation_count, 2)

    def test_keeps_two_independent_tracks_across_frames(self):
        tracker = MultiFaceTracker(SETTINGS)

        first = tracker.update(
            _frame(
                0,
                0,
                _face(0, 0.10, 0.20, 0.25, 0.50),
                _face(1, 0.70, 0.20, 0.85, 0.50),
            )
        )
        second = tracker.update(
            _frame(
                10,
                500,
                _face(0, 0.15, 0.20, 0.30, 0.50),
                _face(1, 0.65, 0.20, 0.80, 0.50),
            )
        )

        self.assertEqual([face.track_id for face in first.faces], [1, 2])
        self.assertTrue(all(face.is_new_track for face in first.faces))
        self.assertEqual([face.track_id for face in second.faces], [1, 2])
        self.assertTrue(all(not face.is_new_track for face in second.faces))
        self.assertEqual(tracker.summaries()[0].observation_count, 2)
        self.assertEqual(tracker.summaries()[1].observation_count, 2)

    def test_recovers_a_track_after_a_short_disappearance(self):
        tracker = MultiFaceTracker(SETTINGS)
        tracker.update(_frame(0, 0, _face(0, 0.1, 0.2, 0.3, 0.5)))
        tracker.update(_frame(10, 500))
        returned = tracker.update(
            _frame(20, 1000, _face(0, 0.12, 0.2, 0.32, 0.5))
        )

        self.assertEqual(returned.faces[0].track_id, 1)
        self.assertFalse(returned.faces[0].is_new_track)
        self.assertEqual(tracker.summaries()[0].observation_count, 2)

    def test_appearance_keeps_the_same_face_on_the_same_track(self):
        tracker = MultiFaceTracker(SETTINGS)
        embedding = np.zeros(512, dtype=np.float32)
        embedding[0] = 1.0
        first = tracker.update(
            _frame(0, 0, _face(0, 0.1, 0.2, 0.3, 0.5, embedding=embedding))
        )
        second = tracker.update(
            _frame(5, 167, _face(0, 0.1, 0.2, 0.3, 0.5, embedding=embedding))
        )

        self.assertEqual(first.faces[0].track_id, second.faces[0].track_id)
        self.assertFalse(second.faces[0].is_new_track)

    def test_appearance_splits_different_faces_at_the_same_position(self):
        tracker = MultiFaceTracker(SETTINGS)
        first_embedding = np.zeros(512, dtype=np.float32)
        first_embedding[0] = 1.0
        second_embedding = np.zeros(512, dtype=np.float32)
        second_embedding[1] = 1.0
        first = tracker.update(
            _frame(
                0,
                0,
                _face(0, 0.1, 0.2, 0.3, 0.5, embedding=first_embedding),
            )
        )
        second = tracker.update(
            _frame(
                5,
                167,
                _face(0, 0.1, 0.2, 0.3, 0.5, embedding=second_embedding),
            )
        )

        self.assertNotEqual(first.faces[0].track_id, second.faces[0].track_id)
        self.assertTrue(second.faces[0].is_new_track)

    def test_creates_a_new_track_after_the_gap_limit(self):
        tracker = MultiFaceTracker(SETTINGS)
        tracker.update(_frame(0, 0, _face(0, 0.1, 0.2, 0.3, 0.5)))
        returned = tracker.update(
            _frame(40, 2000, _face(0, 0.1, 0.2, 0.3, 0.5))
        )

        self.assertEqual(returned.faces[0].track_id, 2)
        self.assertTrue(returned.faces[0].is_new_track)

    def test_motion_prediction_avoids_swapping_crossing_tracks(self):
        tracker = MultiFaceTracker(SETTINGS)
        tracker.update(
            _frame(
                0,
                0,
                _face(0, 0.10, 0.2, 0.25, 0.5),
                _face(1, 0.70, 0.2, 0.85, 0.5),
            )
        )
        tracker.update(
            _frame(
                10,
                500,
                _face(0, 0.28, 0.2, 0.43, 0.5),
                _face(1, 0.52, 0.2, 0.67, 0.5),
            )
        )
        crossed = tracker.update(
            _frame(
                20,
                1000,
                _face(0, 0.30, 0.2, 0.45, 0.5),
                _face(1, 0.50, 0.2, 0.65, 0.5),
            )
        )

        self.assertEqual(crossed.faces[0].track_id, 2)
        self.assertEqual(crossed.faces[1].track_id, 1)

    def test_rejects_out_of_order_frames(self):
        tracker = MultiFaceTracker(SETTINGS)
        tracker.update(_frame(10, 500))
        with self.assertRaises(VideoTrackingError):
            tracker.update(_frame(10, 600))

    def test_track_video_faces_returns_detection_and_track_summary(self):
        detection_summary = VideoDetectionSummary(
            source_fps=10.0,
            source_frame_count=20,
            decoded_frame_count=20,
            sampled_frame_count=2,
            frames_with_faces=2,
            detected_face_count=2,
            width=1000,
            height=1000,
            last_timestamp_ms=1900,
        )

        def fake_detect(object_path, sample_fps, frame_handler):
            self.assertEqual(object_path, "videos/example/source.mp4")
            self.assertEqual(sample_fps, 2.0)
            frame_handler(_frame(0, 0, _face(0, 0.1, 0.2, 0.3, 0.5)))
            frame_handler(_frame(10, 500, _face(0, 0.12, 0.2, 0.32, 0.5)))
            return detection_summary

        frames = []
        with patch("app.video_tracking.detect_video_faces", side_effect=fake_detect):
            summary = track_video_faces(
                "videos/example/source.mp4",
                2.0,
                frame_handler=frames.append,
                settings=SETTINGS,
            )

        self.assertIs(summary.detection, detection_summary)
        self.assertEqual(summary.unique_track_count, 1)
        self.assertEqual(summary.tracks[0].observation_count, 2)
        self.assertEqual([frame.faces[0].track_id for frame in frames], [1, 1])

    def test_hard_scene_cut_starts_a_new_track_at_the_same_position(self):
        detection_summary = VideoDetectionSummary(
            source_fps=30.0,
            source_frame_count=60,
            decoded_frame_count=60,
            sampled_frame_count=2,
            frames_with_faces=2,
            detected_face_count=2,
            width=1000,
            height=1000,
            last_timestamp_ms=167,
        )

        def fake_detect(object_path, sample_fps, frame_handler):
            frame_handler(
                _scene_frame(
                    0, 0, (0, 0, 255), _face(0, 0.30, 0.20, 0.55, 0.55)
                )
            )
            frame_handler(
                _scene_frame(
                    5, 167, (255, 0, 0), _face(0, 0.30, 0.20, 0.55, 0.55)
                )
            )
            return detection_summary

        frames = []
        with patch("app.video_tracking.detect_video_faces", side_effect=fake_detect):
            summary = track_video_faces(
                "videos/example/source.mp4",
                6.0,
                frame_handler=frames.append,
                settings=SETTINGS,
            )

        self.assertEqual([frame.faces[0].track_id for frame in frames], [1, 2])
        self.assertEqual(summary.unique_track_count, 2)

    def test_small_visual_change_keeps_the_active_track(self):
        detection_summary = VideoDetectionSummary(
            source_fps=30.0,
            source_frame_count=60,
            decoded_frame_count=60,
            sampled_frame_count=2,
            frames_with_faces=2,
            detected_face_count=2,
            width=1000,
            height=1000,
            last_timestamp_ms=167,
        )

        def fake_detect(object_path, sample_fps, frame_handler):
            frame_handler(
                _scene_frame(
                    0, 0, (0, 0, 255), _face(0, 0.30, 0.20, 0.55, 0.55)
                )
            )
            frame_handler(
                _scene_frame(
                    5, 167, (0, 0, 240), _face(0, 0.31, 0.20, 0.56, 0.55)
                )
            )
            return detection_summary

        frames = []
        with patch("app.video_tracking.detect_video_faces", side_effect=fake_detect):
            summary = track_video_faces(
                "videos/example/source.mp4",
                6.0,
                frame_handler=frames.append,
                settings=SETTINGS,
            )

        self.assertEqual([frame.faces[0].track_id for frame in frames], [1, 1])
        self.assertEqual(summary.unique_track_count, 1)


if __name__ == "__main__":
    unittest.main()
