import unittest
from types import SimpleNamespace
from uuid import UUID

from app.video_recognition import RecognizedVideoTrack
from benchmark.video_regression_snapshot import _case_payload


KNOWN_ID = UUID("11111111-1111-4111-8111-111111111111")
ANONYMOUS_ID = UUID("22222222-2222-4222-8222-222222222222")


def _track(face_id, status, start_ms, end_ms, name=None):
    return RecognizedVideoTrack(
        track_id=1,
        face_id=face_id,
        status=status,
        recognized=status == "known",
        similarity=0.8,
        threshold=0.45,
        person_id=1 if status == "known" else None,
        name=name,
        metadata=None,
        matched_image_path=None,
        representative_frame_number=10,
        representative_timestamp_ms=start_ms,
        detection_confidence=0.9,
        first_seen_ms=start_ms,
        last_seen_ms=end_ms,
        observation_count=5,
    )


class VideoRegressionSnapshotTests(unittest.TestCase):
    def test_groups_known_and_anonymous_tracks_into_case(self):
        job = SimpleNamespace(
            process_id=UUID("33333333-3333-4333-8333-333333333333"),
            original_filename="test.mp4",
        )
        tracks = (
            _track(KNOWN_ID, "known", 0, 2000, "Test Person"),
            _track(KNOWN_ID, "known", 4000, 5000, "Test Person"),
            _track(ANONYMOUS_ID, "anonymous", 2500, 2800),
        )

        case = _case_payload(1, job, tracks)

        self.assertEqual(case["source"]["filename"], "test.mp4")
        self.assertEqual(len(case["expectedFaces"]), 1)
        self.assertEqual(len(case["expectedFaces"][0]["intervals"]), 2)
        self.assertEqual(len(case["expectedAnonymousFaces"]), 1)
        self.assertEqual(case["limits"]["minimumAnonymousTracks"], 1)
        self.assertEqual(case["limits"]["maximumTotalTracks"], 3)
        self.assertEqual(case["limits"]["maximumShortTracks"], 1)


if __name__ == "__main__":
    unittest.main()
