import json
import tempfile
import unittest
from pathlib import Path
from uuid import UUID

from benchmark.video_acceptance import (
    PredictedTrack,
    evaluate_case,
    load_manifest,
)
from benchmark.video_acceptance_report import write_video_acceptance_reports


KNOWN_FACE_ID = UUID("11111111-1111-4111-8111-111111111111")
WRONG_FACE_ID = UUID("22222222-2222-4222-8222-222222222222")
ANONYMOUS_FACE_ID = UUID("33333333-3333-4333-8333-333333333333")


def _manifest_payload():
    return {
        "version": 1,
        "suiteName": "Test Suite",
        "sampleFps": 6,
        "defaults": {
            "timeToleranceSeconds": 1,
            "minimumIdentityRecall": 1,
            "minimumTemporalIoU": 0.5,
            "maximumUnexpectedKnown": 0,
            "minimumAnonymousRecall": 1,
            "minimumAnonymousTemporalIoU": 0.5,
            "minimumAnonymousTracks": 1,
            "maximumAnonymousTracks": 1,
            "maximumTracksPerExpectedFace": 2,
            "maximumTotalTracks": 3,
            "maximumShortTracks": 0,
            "maximumRealtimeFactor": 1,
        },
        "cases": [
            {
                "id": "known-person",
                "source": {"filename": "known.mp4"},
                "evaluationWindow": {"startSeconds": 0, "endSeconds": 30},
                "expectedFaces": [
                    {
                        "faceId": str(KNOWN_FACE_ID),
                        "name": "Known Person",
                        "intervals": [
                            {"startSeconds": 0, "endSeconds": 5},
                            {"startSeconds": 20, "endSeconds": 30},
                        ],
                    }
                ],
                "expectedAnonymousFaces": [
                    {
                        "label": "Visitor",
                        "intervals": [
                            {"startSeconds": 10, "endSeconds": 12},
                        ],
                    }
                ],
            }
        ],
    }


def _load(payload):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "manifest.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return load_manifest(path)


class VideoAcceptanceTests(unittest.TestCase):
    def test_loads_manifest_and_merges_limits(self):
        manifest = _load(_manifest_payload())

        self.assertEqual(manifest.sample_fps, 6)
        self.assertEqual(len(manifest.cases), 1)
        case = manifest.cases[0]
        self.assertEqual(case.filename, "known.mp4")
        self.assertEqual(case.limits.time_tolerance_ms, 1000)
        self.assertEqual(case.limits.maximum_anonymous_tracks, 1)
        self.assertEqual(len(case.expected_faces[0].intervals), 2)
        self.assertEqual(case.evaluation_window.start_ms, 0)
        self.assertEqual(case.evaluation_window.end_ms, 30000)
        self.assertEqual(case.expected_anonymous_faces[0].label, "Visitor")

    def test_rejects_duplicate_case_ids(self):
        payload = _manifest_payload()
        payload["cases"].append(dict(payload["cases"][0]))

        with self.assertRaisesRegex(ValueError, "Tekrarlanan case id"):
            _load(payload)

    def test_passes_matching_identity_and_tolerant_intervals(self):
        case = _load(_manifest_payload()).cases[0]
        tracks = (
            PredictedTrack(
                face_id=KNOWN_FACE_ID,
                status="known",
                name="Known Person",
                first_seen_ms=500,
                last_seen_ms=5500,
                confidence=0.8,
                track_id=1,
                source_track_id=10,
                observation_count=18,
                detection_confidence=0.91,
                threshold=0.45,
            ),
            PredictedTrack(
                face_id=KNOWN_FACE_ID,
                status="known",
                name="Known Person",
                first_seen_ms=19500,
                last_seen_ms=30500,
                confidence=0.82,
                track_id=2,
                source_track_id=10,
                observation_count=24,
                detection_confidence=0.93,
                threshold=0.45,
            ),
            PredictedTrack(
                face_id=ANONYMOUS_FACE_ID,
                status="new_anonymous",
                name=None,
                first_seen_ms=10000,
                last_seen_ms=12000,
                confidence=None,
            ),
        )

        result = evaluate_case(case, tracks, 10, 30)

        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["metrics"]["identityRecall"], 1.0)
        self.assertEqual(result["metrics"]["temporalIou"], 1.0)
        self.assertEqual(result["metrics"]["anonymousTrackCount"], 1)
        self.assertEqual(result["metrics"]["anonymousRecall"], 1.0)
        self.assertEqual(result["metrics"]["anonymousTemporalIou"], 1.0)
        self.assertEqual(result["metrics"]["fragmentedIdentityCount"], 1)
        self.assertEqual(result["trackDiagnostics"][0]["trackId"], 1)
        self.assertEqual(result["identityDiagnostics"][0]["trackCount"], 2)

    def test_marks_short_low_margin_and_unexpected_known_tracks(self):
        case = _load(_manifest_payload()).cases[0]
        tracks = (
            PredictedTrack(
                face_id=WRONG_FACE_ID,
                status="known",
                name="Wrong Person",
                first_seen_ms=2000,
                last_seen_ms=2500,
                confidence=0.49,
                detection_confidence=0.42,
                threshold=0.45,
            ),
        )

        result = evaluate_case(case, tracks, 3, 30)

        flags = result["trackDiagnostics"][0]["flags"]
        self.assertIn("short_track", flags)
        self.assertIn("low_known_margin", flags)
        self.assertIn("low_detection_confidence", flags)
        self.assertIn("unexpected_known_identity", flags)
        self.assertEqual(result["metrics"]["shortTrackCount"], 1)
        self.assertEqual(result["metrics"]["lowConfidenceKnownTrackCount"], 1)

    def test_fails_missed_identity_and_unexpected_known_face(self):
        case = _load(_manifest_payload()).cases[0]
        tracks = (
            PredictedTrack(
                face_id=WRONG_FACE_ID,
                status="known",
                name="Wrong Person",
                first_seen_ms=0,
                last_seen_ms=5000,
                confidence=0.7,
            ),
        )

        result = evaluate_case(case, tracks, 40, 30)

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["metrics"]["identityRecall"], 0.0)
        self.assertEqual(result["metrics"]["unexpectedKnownCount"], 1)
        self.assertIn(str(KNOWN_FACE_ID), result["missedFaceIds"])

    def test_fails_when_a_labeled_anonymous_face_is_not_found(self):
        case = _load(_manifest_payload()).cases[0]
        tracks = (
            PredictedTrack(
                face_id=KNOWN_FACE_ID,
                status="known",
                name="Known Person",
                first_seen_ms=0,
                last_seen_ms=5000,
                confidence=0.8,
            ),
        )

        result = evaluate_case(case, tracks, 5, 30)

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["metrics"]["anonymousRecall"], 0.0)
        self.assertFalse(result["anonymousFaces"][0]["matched"])

    def test_ignores_tracks_outside_the_evaluation_window(self):
        case = _load(_manifest_payload()).cases[0]
        tracks = (
            PredictedTrack(
                face_id=WRONG_FACE_ID,
                status="known",
                name="Outside Person",
                first_seen_ms=40000,
                last_seen_ms=45000,
                confidence=0.9,
            ),
        )

        result = evaluate_case(case, tracks, 5, 60)

        self.assertEqual(result["metrics"]["totalTrackCount"], 0)
        self.assertEqual(result["metrics"]["unexpectedKnownCount"], 0)

    def test_writes_json_csv_and_html_reports(self):
        result = {
            "suiteName": "Test Suite",
            "createdAt": "2026-08-27T00:00:00+00:00",
            "status": "passed",
            "summary": {
                "executedCases": 1,
                "passedCases": 1,
                "failedCases": 0,
            },
            "cases": [
                {
                    "id": "case-1",
                    "status": "passed",
                    "source": {"filename": "test.mp4"},
                    "metrics": {
                        "identityRecall": 1.0,
                        "temporalIou": 0.9,
                        "unexpectedKnownCount": 0,
                        "anonymousTrackCount": 0,
                        "totalTrackCount": 1,
                        "modelWarmupSeconds": 3,
                        "processingSeconds": 2,
                        "endToEndSeconds": 5,
                        "realtimeFactor": 0.2,
                    },
                    "checks": [
                        {
                            "name": "identity_recall",
                            "actual": 1.0,
                            "expected": ">=1.0",
                            "passed": True,
                        }
                    ],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            paths = write_video_acceptance_reports(
                Path(directory) / "run",
                result,
            )
            self.assertTrue(all(Path(path).is_file() for path in paths.values()))
            self.assertIn("case-1", Path(paths["html"]).read_text(encoding="utf-8"))
            csv_report = Path(paths["csv"]).read_text(encoding="utf-8")
            self.assertIn("model_warmup_seconds", csv_report)
            self.assertIn("end_to_end_seconds", csv_report)


if __name__ == "__main__":
    unittest.main()
