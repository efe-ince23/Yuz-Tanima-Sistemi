import unittest

from app.models import (
    RecognitionProcess,
    VideoAppearanceSegment,
    VideoFaceObservation,
    VideoJob,
    VideoTrack,
)
from app.process_tracking import TRACKED_FACE_PATHS


class VideoModelContractTests(unittest.TestCase):
    def test_video_upload_has_a_tracked_process(self) -> None:
        self.assertEqual(TRACKED_FACE_PATHS["/api/videos"], "video_recognize")
        self.assertEqual(
            TRACKED_FACE_PATHS["/api/videos/live-recordings"],
            "video_recognize",
        )

    def test_video_tables_are_linked_to_the_process(self):
        process_foreign_keys = {
            foreign_key.target_fullname: foreign_key.ondelete
            for foreign_key in VideoJob.__table__.foreign_keys
        }
        self.assertEqual(
            process_foreign_keys["recognition_processes.process_id"], "CASCADE"
        )
        self.assertIn("video_job", RecognitionProcess.__mapper__.relationships)

    def test_video_child_tables_have_cascade_relationships(self):
        for model in (VideoTrack, VideoFaceObservation, VideoAppearanceSegment):
            foreign_keys = {
                foreign_key.target_fullname: foreign_key.ondelete
                for foreign_key in model.__table__.foreign_keys
            }
            self.assertIn("video_jobs.process_id", foreign_keys)
            self.assertEqual(foreign_keys["video_jobs.process_id"], "CASCADE")

    def test_observation_contract_keeps_time_and_normalized_box(self):
        columns = VideoFaceObservation.__table__.columns
        for column_name in (
            "frame_number",
            "timestamp_ms",
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
            "detection_confidence",
            "recognition_confidence",
        ):
            self.assertIn(column_name, columns)


if __name__ == "__main__":
    unittest.main()
