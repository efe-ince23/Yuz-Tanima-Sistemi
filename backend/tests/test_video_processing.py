import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch
from uuid import UUID

from app.models import (
    RecognitionProcess,
    VideoAppearanceSegment,
    VideoFaceObservation,
    VideoJob,
    VideoTrack,
)
from app.video_detection import VideoDetectionSummary
from app.video_processing import (
    _Observation,
    _appearance_groups,
    _normalize_live_job,
    _persist_live_manifest,
    _persist_result,
)
from app.video_recognition import RecognizedVideoTrack, VideoRecognitionSummary
from app.video_tracking import VideoTrackSummary, VideoTrackingSummary
from app.video_upload import ValidatedVideo, VideoMetadata


class VideoResultPersistenceTests(unittest.TestCase):
    def test_rejects_an_incomplete_live_manifest_for_full_analysis_fallback(self):
        process_id = UUID("12121212-1212-4212-8212-121212121212")
        process = RecognitionProcess(
            process_id=process_id,
            operation_type="video_recognize",
            status="processing",
        )
        job = VideoJob(
            process_id=process_id,
            status="processing",
            original_filename="eksik-canli.mp4",
            object_path=f"videos/{process_id}/source.mp4",
            content_type="video/mp4",
            file_size_bytes=100,
        )
        session = MagicMock()
        manifest = {
            "duration_ms": 10000,
            "analysis_count": 1,
            "first_analysis_ms": 0,
            "last_analysis_ms": 500,
            "observations": [{"face_id": str(process_id)}],
        }

        used = _persist_live_manifest(session, job, process, manifest)

        self.assertFalse(used)
        session.add.assert_not_called()
        self.assertEqual(job.status, "processing")

    def test_persists_live_observations_without_reanalyzing_the_video(self):
        process_id = UUID("abababab-abab-4bab-8bab-abababababab")
        face_id = UUID("cdcdcdcd-cdcd-4dcd-8dcd-cdcdcdcdcdcd")
        process = RecognitionProcess(
            process_id=process_id,
            operation_type="video_recognize",
            status="processing",
        )
        job = VideoJob(
            process_id=process_id,
            status="processing",
            original_filename="canli.mp4",
            object_path=f"videos/{process_id}/source.mp4",
            content_type="video/mp4",
            file_size_bytes=100,
            duration_seconds=5.0,
            source_fps=25.0,
        )
        session = MagicMock()

        def assign_track_id():
            for call in session.add.call_args_list:
                item = call.args[0]
                if isinstance(item, VideoTrack) and item.id is None:
                    item.id = 1

        session.flush.side_effect = assign_track_id
        manifest = {
            "duration_ms": 5000,
            "analysis_count": 3,
            "first_analysis_ms": 100,
            "last_analysis_ms": 4000,
            "observations": [
                {
                    "timestamp_ms": timestamp,
                    "face_id": str(face_id),
                    "status": "known",
                    "name": "Test Kisi",
                    "metadata": {"description": "Canli test"},
                    "bounding_box": {"x1": 0.1, "y1": 0.2, "x2": 0.4, "y2": 0.6},
                    "detection_confidence": 0.95,
                    "recognition_confidence": 0.88,
                    "matched_image_url": "/media/persons/1/test.jpg",
                }
                for timestamp in (100, 1000, 4000)
            ],
        }

        used = _persist_live_manifest(session, job, process, manifest)

        self.assertTrue(used)
        self.assertEqual(job.status, "completed")
        self.assertEqual(job.sampled_frame_count, 3)
        self.assertEqual(job.detected_face_count, 3)
        self.assertEqual(job.unique_face_count, 1)
        self.assertEqual(process.task_detail["stage"], "live_manifest_completed")
        tracks = [
            call.args[0]
            for call in session.add.call_args_list
            if isinstance(call.args[0], VideoTrack)
        ]
        segments = [
            call.args[0]
            for call in session.add.call_args_list
            if isinstance(call.args[0], VideoAppearanceSegment)
        ]
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].best_image_path, "persons/1/test.jpg")
        self.assertEqual(len(segments), 3)
        observations = session.add_all.call_args.args[0]
        self.assertEqual(len(observations), 3)

    def test_normalizes_a_queued_live_recording_before_face_analysis(self):
        process_id = UUID("eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee")
        process = RecognitionProcess(
            process_id=process_id,
            operation_type="video_recognize",
            status="processing",
        )
        job = VideoJob(
            process_id=process_id,
            status="processing",
            original_filename="canli-kamera.mp4",
            object_path=f"videos/{process_id}/source.webm",
            content_type="video/webm",
            file_size_bytes=80,
        )
        session = MagicMock()

        with TemporaryDirectory() as directory:
            normalized_path = Path(directory) / "normalized.mp4"
            normalized_path.write_bytes(b"normalized-video")
            normalized = ValidatedVideo(
                temporary_path=normalized_path,
                original_filename="canli-kamera.mp4",
                content_type="video/mp4",
                file_size_bytes=16,
                metadata=VideoMetadata("mp4", "h264", 2.0, 25.0, 640, 360, 50),
            )

            def download(_object_path, destination):
                destination.write_bytes(b"\x1a\x45\xdf\xa3webm")
                return destination

            with patch("app.video_processing.download_object_to_file", side_effect=download), patch(
                "app.video_processing.normalize_live_recording",
                return_value=normalized,
            ), patch("app.video_processing.save_file_object") as save, patch(
                "app.video_processing.delete_face_image",
            ) as delete:
                _normalize_live_job(session, job, process)

        self.assertEqual(job.content_type, "video/mp4")
        self.assertTrue(job.object_path.endswith("/source.mp4"))
        self.assertEqual(job.duration_seconds, 2.0)
        self.assertEqual(job.source_fps, 25.0)
        self.assertEqual(job.progress_percent, 2.0)
        self.assertEqual(process.task_detail["stage"], "face_analysis")
        save.assert_called_once()
        delete.assert_called_once_with(f"videos/{process_id}/source.webm")
        session.commit.assert_called_once()

    def test_persists_tracks_observations_and_completed_process(self):
        process_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
        face_id = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
        process = RecognitionProcess(
            process_id=process_id,
            operation_type="video_recognize",
            status="processing",
        )
        job = VideoJob(
            process_id=process_id,
            status="processing",
            original_filename="test.mp4",
            object_path="videos/test/source.mp4",
            content_type="video/mp4",
            file_size_bytes=100,
        )
        tracking = VideoTrackingSummary(
            detection=VideoDetectionSummary(
                source_fps=10.0,
                source_frame_count=20,
                decoded_frame_count=20,
                sampled_frame_count=2,
                frames_with_faces=2,
                detected_face_count=2,
                width=100,
                height=100,
                last_timestamp_ms=1000,
            ),
            unique_track_count=1,
            tracks=(
                VideoTrackSummary(
                    track_id=1,
                    first_seen_ms=0,
                    last_seen_ms=1000,
                    first_frame_number=0,
                    last_frame_number=10,
                    observation_count=2,
                    best_confidence=0.94,
                    best_frame_number=10,
                ),
            ),
        )
        summary = VideoRecognitionSummary(
            tracking=tracking,
            tracks=(
                RecognizedVideoTrack(
                    track_id=1,
                    face_id=face_id,
                    status="known",
                    recognized=True,
                    similarity=0.88,
                    threshold=0.45,
                    person_id=4,
                    name="Test Kisi",
                    metadata={"description": "Test"},
                    matched_image_path="persons/4/test.jpg",
                    representative_frame_number=10,
                    representative_timestamp_ms=1000,
                    detection_confidence=0.94,
                ),
            ),
        )
        observations = [
            _Observation(1, 0, 0, (0.1, 0.2, 0.4, 0.6), 0.90),
            _Observation(1, 10, 1000, (0.2, 0.2, 0.5, 0.6), 0.94),
        ]
        session = MagicMock()

        def assign_track_id(_objects=None):
            added = [
                call.args[0]
                for call in session.add.call_args_list
                if isinstance(call.args[0], VideoTrack)
            ]
            for index, track in enumerate(added, start=1):
                track.id = index

        session.flush.side_effect = assign_track_id

        _persist_result(session, job, process, summary, observations)

        self.assertEqual(job.status, "completed")
        self.assertEqual(job.sampled_frame_count, 2)
        self.assertEqual(job.detected_face_count, 2)
        self.assertEqual(job.unique_face_count, 1)
        self.assertEqual(process.status, "completed")
        self.assertEqual(process.face_count, 1)
        persisted_rows = session.add_all.call_args.args[0]
        persisted_observations = [
            item for item in persisted_rows if isinstance(item, VideoFaceObservation)
        ]
        persisted_segments = [
            item for item in persisted_rows if isinstance(item, VideoAppearanceSegment)
        ]
        self.assertEqual(len(persisted_observations), 2)
        self.assertTrue(
            all(isinstance(item, VideoFaceObservation) for item in persisted_observations)
        )
        self.assertTrue(all(item.face_id == face_id for item in persisted_observations))
        self.assertEqual(len(persisted_segments), 2)
        self.assertEqual(persisted_segments[0].start_ms, 0)
        self.assertEqual(persisted_segments[0].end_ms, 0)
        self.assertEqual(persisted_segments[0].observation_count, 1)
        self.assertEqual(persisted_segments[1].start_ms, 1000)
        self.assertEqual(persisted_segments[1].end_ms, 1000)
        self.assertEqual(persisted_segments[1].observation_count, 1)
        self.assertEqual(process.result["tracks"][0]["firstSeenMs"], 0)
        self.assertEqual(process.result["tracks"][0]["lastSeenMs"], 1000)

    def test_appearance_groups_split_a_long_absence(self):
        observations = [
            _Observation(1, 0, 0, (0.1, 0.1, 0.2, 0.2), 0.9),
            _Observation(1, 3, 300, (0.1, 0.1, 0.2, 0.2), 0.9),
            _Observation(1, 30, 3000, (0.1, 0.1, 0.2, 0.2), 0.9),
        ]

        groups = _appearance_groups(observations, max_gap_ms=1500)

        self.assertEqual([[item.timestamp_ms for item in group] for group in groups], [[0, 300], [3000]])

    def test_persists_identity_segments_from_one_spatial_track_separately(self):
        process_id = UUID("cccccccc-cccc-4ccc-8ccc-cccccccccccc")
        first_face_id = UUID("11111111-1111-4111-8111-111111111111")
        second_face_id = UUID("22222222-2222-4222-8222-222222222222")
        process = RecognitionProcess(
            process_id=process_id,
            operation_type="video_recognize",
            status="processing",
        )
        job = VideoJob(
            process_id=process_id,
            status="processing",
            original_filename="identity-change.mp4",
            object_path="videos/identity-change/source.mp4",
            content_type="video/mp4",
            file_size_bytes=100,
        )
        tracking = VideoTrackingSummary(
            detection=VideoDetectionSummary(
                source_fps=10.0,
                source_frame_count=40,
                decoded_frame_count=40,
                sampled_frame_count=4,
                frames_with_faces=4,
                detected_face_count=4,
                width=100,
                height=100,
                last_timestamp_ms=3000,
            ),
            unique_track_count=1,
            tracks=(
                VideoTrackSummary(
                    track_id=1,
                    first_seen_ms=0,
                    last_seen_ms=3000,
                    first_frame_number=0,
                    last_frame_number=30,
                    observation_count=4,
                    best_confidence=0.94,
                    best_frame_number=30,
                ),
            ),
        )

        def recognized(track_id, face_id, first_seen_ms, last_seen_ms):
            return RecognizedVideoTrack(
                track_id=track_id,
                face_id=face_id,
                status="known",
                recognized=True,
                similarity=0.8,
                threshold=0.45,
                person_id=track_id,
                name=f"Kisi {track_id}",
                metadata=None,
                matched_image_path=None,
                representative_frame_number=first_seen_ms // 100,
                representative_timestamp_ms=first_seen_ms,
                detection_confidence=0.9,
                source_track_id=1,
                first_seen_ms=first_seen_ms,
                last_seen_ms=last_seen_ms,
                observation_count=2,
            )

        summary = VideoRecognitionSummary(
            tracking=tracking,
            tracks=(
                recognized(1, first_face_id, 0, 1000),
                recognized(2, second_face_id, 2000, 3000),
            ),
        )
        observations = [
            _Observation(1, 0, 0, (0.1, 0.1, 0.3, 0.4), 0.9),
            _Observation(1, 10, 1000, (0.1, 0.1, 0.3, 0.4), 0.9),
            _Observation(1, 20, 2000, (0.5, 0.1, 0.7, 0.4), 0.9),
            _Observation(1, 30, 3000, (0.5, 0.1, 0.7, 0.4), 0.9),
        ]
        session = MagicMock()

        def assign_track_ids(_objects=None):
            tracks = [
                call.args[0]
                for call in session.add.call_args_list
                if isinstance(call.args[0], VideoTrack)
            ]
            for index, track in enumerate(tracks, start=1):
                track.id = index

        session.flush.side_effect = assign_track_ids

        _persist_result(session, job, process, summary, observations)

        persisted_tracks = [
            call.args[0]
            for call in session.add.call_args_list
            if isinstance(call.args[0], VideoTrack)
        ]
        rows = session.add_all.call_args.args[0]
        persisted_observations = [
            item for item in rows if isinstance(item, VideoFaceObservation)
        ]
        self.assertEqual(len(persisted_tracks), 2)
        self.assertEqual(
            [(track.first_seen_ms, track.last_seen_ms) for track in persisted_tracks],
            [(0, 1000), (2000, 3000)],
        )
        self.assertEqual(
            [item.face_id for item in persisted_observations],
            [first_face_id, first_face_id, second_face_id, second_face_id],
        )
        self.assertEqual(job.unique_face_count, 2)


if __name__ == "__main__":
    unittest.main()
