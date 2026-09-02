import unittest
from types import SimpleNamespace
from uuid import UUID

from app.routers.videos import _video_track_result


def observation(timestamp_ms, box):
    return SimpleNamespace(
        frame_number=timestamp_ms // 100,
        timestamp_ms=timestamp_ms,
        bbox_x1=box[0],
        bbox_y1=box[1],
        bbox_x2=box[2],
        bbox_y2=box[3],
        detection_confidence=0.9,
        recognition_confidence=0.8,
    )


def track(track_number, face_id, box):
    return SimpleNamespace(
        track_number=track_number,
        face_id=face_id,
        face_status="known",
        first_seen_ms=0,
        last_seen_ms=1000,
        observation_count=2,
        best_detection_confidence=0.9,
        best_recognition_confidence=0.8,
        best_frame_number=0,
        best_image_path=None,
        appearance_segments=[],
        observations=[observation(0, box), observation(1000, box)],
    )


class VideoResultTrackTests(unittest.TestCase):
    def test_same_face_id_keeps_two_independent_tracks_and_boxes(self):
        face_id = UUID("cdcdcdcd-cdcd-4dcd-8dcd-cdcdcdcdcdcd")
        person = SimpleNamespace(
            first_name="Ayni",
            last_name="Kisi",
            description=None,
        )

        first = _video_track_result(
            track(1, face_id, (0.1, 0.1, 0.3, 0.4)),
            person,
        )
        second = _video_track_result(
            track(2, face_id, (0.6, 0.1, 0.8, 0.4)),
            person,
        )

        self.assertEqual(first.face_id, second.face_id)
        self.assertEqual([first.track_id, second.track_id], [1, 2])
        self.assertNotEqual(
            first.observations[0].bounding_box,
            second.observations[0].bounding_box,
        )
        self.assertEqual([first.name, second.name], ["Ayni Kisi", "Ayni Kisi"])


if __name__ == "__main__":
    unittest.main()
