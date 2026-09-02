import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import UUID

import numpy as np

from app.video_detection import VideoDetectionSummary, VideoFaceDetection
from app.video_config import get_video_recognition_settings
from app.video_recognition import _frontal_score, recognize_video_tracks
from app.video_tracking import (
    TrackedVideoFace,
    TrackedVideoFrame,
    VideoTrackSummary,
    VideoTrackingSummary,
)


def _detection(face_index, box, confidence=0.9):
    x1, y1, x2, y2 = box
    return VideoFaceDetection(
        face_index=face_index,
        confidence=confidence,
        bounding_box=(x1, y1, x2, y2),
        normalized_bounding_box=(
            x1 / 100,
            y1 / 100,
            x2 / 100,
            y2 / 100,
        ),
        landmarks=((x1, y1), (x2, y1), ((x1 + x2) / 2, (y1 + y2) / 2)),
    )


def _tracked_frame(frame_number, timestamp_ms, faces):
    return TrackedVideoFrame(
        frame_number=frame_number,
        timestamp_ms=timestamp_ms,
        width=100,
        height=100,
        image=np.full((100, 100, 3), frame_number + 1, dtype=np.uint8),
        faces=tuple(faces),
    )


def _textured_tracked_frame(frame_number, timestamp_ms, faces):
    pattern = ((np.indices((100, 100)).sum(axis=0) % 2) * 255).astype(np.uint8)
    return TrackedVideoFrame(
        frame_number=frame_number,
        timestamp_ms=timestamp_ms,
        width=100,
        height=100,
        image=np.repeat(pattern[:, :, None], 3, axis=2),
        faces=tuple(faces),
    )


def _tracking_summary(track_count, detected_face_count, observation_count=2):
    return VideoTrackingSummary(
        detection=VideoDetectionSummary(
            source_fps=10.0,
            source_frame_count=20,
            decoded_frame_count=20,
            sampled_frame_count=2,
            frames_with_faces=2 if detected_face_count else 0,
            detected_face_count=detected_face_count,
            width=100,
            height=100,
            last_timestamp_ms=1900,
        ),
        unique_track_count=track_count,
        tracks=tuple(
            VideoTrackSummary(
                track_id=index,
                first_seen_ms=0,
                last_seen_ms=1000,
                first_frame_number=0,
                last_frame_number=10,
                observation_count=observation_count,
                best_confidence=0.9,
                best_frame_number=0,
            )
            for index in range(1, track_count + 1)
        ),
    )


def _legacy_anonymous_settings():
    return replace(
        get_video_recognition_settings(),
        anonymous_min_quality_samples=2,
        anonymous_min_duration_ms=1,
        anonymous_min_detection_confidence=0.0,
        anonymous_min_face_size_px=1,
        anonymous_min_sharpness=0.0,
        anonymous_min_embedding_consistency=0.0,
    )


class _FakeRecognizer:
    def __init__(self, embeddings):
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.batch_sizes = []

    def align(self, image, face):
        del image
        return np.full((112, 112, 3), int(face.bbox[0]), dtype=np.uint8)

    def embed_aligned(self, aligned_faces):
        self.batch_sizes.append(len(aligned_faces))
        return self.embeddings


class VideoRecognitionTests(unittest.TestCase):
    def _run_tracking(self, frames, summary):
        def fake_tracking(object_path, sample_fps, frame_handler):
            self.assertEqual(object_path, "videos/test/source.mp4")
            self.assertEqual(sample_fps, 3.0)
            for frame in frames:
                frame_handler(frame)
            return summary

        return fake_tracking

    def test_frontal_score_accepts_a_rotated_frontal_face(self):
        landmarks = np.asarray(
            ((-1.0, 0.0), (1.0, 0.0), (0.0, 1.0), (-0.6, 2.0), (0.6, 2.0)),
            dtype=np.float32,
        )
        angle = np.deg2rad(30.0)
        rotation = np.asarray(
            ((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle))),
            dtype=np.float32,
        )

        self.assertGreaterEqual(_frontal_score(landmarks @ rotation.T), 0.95)

    def test_frontal_score_still_rejects_a_profile_face(self):
        landmarks = np.asarray(
            ((-1.0, 0.0), (1.0, 0.0), (0.9, 1.0), (0.2, 2.0), (1.2, 2.0)),
            dtype=np.float32,
        )

        self.assertLess(_frontal_score(landmarks), 0.35)

    def test_recognizes_two_known_tracks_in_one_arcface_batch(self):
        frames = [
            _tracked_frame(
                0,
                0,
                (
                    TrackedVideoFace(1, True, _detection(0, (5, 5, 35, 45))),
                    TrackedVideoFace(2, True, _detection(1, (55, 5, 90, 45))),
                ),
            )
        ]
        tracking = _tracking_summary(2, 2)
        recognizer = _FakeRecognizer(((1.0, 0.0), (0.0, 1.0)))
        engine = SimpleNamespace(recognizer=recognizer)
        persons = (
            SimpleNamespace(
                id=4,
                face_id=UUID("11111111-1111-1111-1111-111111111111"),
                first_name="Fatih",
                last_name="Terim",
                description="Teknik direktor",
            ),
            SimpleNamespace(
                id=5,
                face_id=UUID("22222222-2222-2222-2222-222222222222"),
                first_name="Burak",
                last_name="Yilmaz",
                description="Futbolcu",
            ),
        )
        matches = [
            (persons[0], SimpleNamespace(image_path="persons/4/a.jpg"), 0.81),
            (persons[1], SimpleNamespace(image_path="persons/5/b.jpg"), 0.76),
        ]
        session = MagicMock()

        with patch(
            "app.video_recognition.track_video_faces",
            side_effect=self._run_tracking(frames, tracking),
        ), patch(
            "app.video_recognition.get_yolo_arcface_engine",
            return_value=engine,
        ), patch(
            "app.video_recognition.lock_anonymous_matching"
        ), patch(
            "app.video_recognition.find_closest_face", side_effect=matches
        ):
            result = recognize_video_tracks(
                session,
                "videos/test/source.mp4",
                3.0,
            )

        self.assertEqual(recognizer.batch_sizes, [2])
        self.assertEqual([track.status for track in result.tracks], ["known", "known"])
        self.assertEqual([track.name for track in result.tracks], ["Fatih Terim", "Burak Yilmaz"])
        self.assertTrue(all(track.recognized for track in result.tracks))
        session.commit.assert_not_called()

    def test_recognizes_same_known_person_on_two_simultaneous_tracks(self):
        frames = [
            _tracked_frame(
                0,
                0,
                (
                    TrackedVideoFace(1, True, _detection(0, (5, 5, 35, 45))),
                    TrackedVideoFace(2, True, _detection(1, (55, 5, 90, 45))),
                ),
            )
        ]
        tracking = _tracking_summary(2, 2)
        recognizer = _FakeRecognizer(((1.0, 0.0), (0.96, 0.28)))
        engine = SimpleNamespace(recognizer=recognizer)
        face_id = UUID("abababab-abab-4bab-8bab-abababababab")
        person = SimpleNamespace(
            id=10,
            face_id=face_id,
            first_name="Ayni",
            last_name="Kisi",
            description=None,
        )
        known_match = (
            person,
            SimpleNamespace(image_path="persons/10/a.jpg"),
            0.82,
        )

        with patch(
            "app.video_recognition.track_video_faces",
            side_effect=self._run_tracking(frames, tracking),
        ), patch(
            "app.video_recognition.get_video_recognition_settings",
            return_value=_legacy_anonymous_settings(),
        ), patch(
            "app.video_recognition.get_yolo_arcface_engine",
            return_value=engine,
        ), patch(
            "app.video_recognition.lock_anonymous_matching"
        ), patch(
            "app.video_recognition.find_closest_face",
            side_effect=[known_match, None],
        ):
            result = recognize_video_tracks(
                MagicMock(),
                "videos/test/source.mp4",
                3.0,
            )

        self.assertEqual(len(result.tracks), 2)
        self.assertEqual([track.track_id for track in result.tracks], [1, 2])
        self.assertEqual({track.face_id for track in result.tracks}, {face_id})
        self.assertTrue(all(track.status == "known" for track in result.tracks))

    def test_continuity_keeps_match_evidence_for_a_separate_segment(self):
        first_face_id = UUID("10101010-1010-4010-8010-101010101010")
        second_face_id = UUID("20202020-2020-4020-8020-202020202020")
        first_person = SimpleNamespace(
            id=10,
            face_id=first_face_id,
            first_name="Birinci",
            last_name="Kisi",
            description=None,
        )
        second_person = SimpleNamespace(
            id=20,
            face_id=second_face_id,
            first_name="Ikinci",
            last_name="Kisi",
            description=None,
        )
        frames = [
            _tracked_frame(
                index,
                timestamp_ms,
                (TrackedVideoFace(1, index == 0, _detection(0, (10, 10, 50, 60))),),
            )
            for index, timestamp_ms in enumerate((0, 3000, 6000))
        ]
        tracking = _tracking_summary(1, 3)
        recognizer = _FakeRecognizer(((1.0, 0.0), (0.0, 1.0), (0.95, 0.31)))
        engine = SimpleNamespace(recognizer=recognizer)
        matches = [
            (
                first_person,
                SimpleNamespace(image_path="persons/10/a.jpg"),
                0.84,
            ),
            (
                second_person,
                SimpleNamespace(image_path="persons/20/a.jpg"),
                0.83,
            ),
            None,
        ]

        with patch(
            "app.video_recognition.track_video_faces",
            side_effect=self._run_tracking(frames, tracking),
        ), patch(
            "app.video_recognition.get_yolo_arcface_engine",
            return_value=engine,
        ), patch(
            "app.video_recognition.lock_anonymous_matching"
        ), patch(
            "app.video_recognition.find_closest_face",
            side_effect=matches,
        ):
            result = recognize_video_tracks(
                MagicMock(),
                "videos/test/source.mp4",
                3.0,
            )

        self.assertEqual([track.status for track in result.tracks], ["known"] * 3)
        self.assertEqual(
            [track.face_id for track in result.tracks],
            [first_face_id, second_face_id, first_face_id],
        )
        self.assertEqual(
            [track.name for track in result.tracks],
            ["Birinci Kisi", "Ikinci Kisi", "Birinci Kisi"],
        )

    def test_uses_multiple_samples_and_majority_for_a_track(self):
        face_id = UUID("99999999-9999-4999-8999-999999999999")
        person = SimpleNamespace(
            id=9,
            face_id=face_id,
            first_name="Test",
            last_name="Kisi",
            description=None,
        )
        frames = [
            _tracked_frame(
                index,
                index * 300,
                (TrackedVideoFace(1, index == 0, _detection(0, (10, 10, 50, 60))),),
            )
            for index in range(3)
        ]
        tracking = _tracking_summary(1, 3, observation_count=3)
        recognizer = _FakeRecognizer(((1.0, 0.0), (0.9, 0.1), (0.0, 1.0)))
        engine = SimpleNamespace(recognizer=recognizer)
        matches = [
            (person, SimpleNamespace(image_path="persons/9/a.jpg"), 0.84),
            (person, SimpleNamespace(image_path="persons/9/a.jpg"), 0.78),
            None,
        ]

        with patch(
            "app.video_recognition.track_video_faces",
            side_effect=self._run_tracking(frames, tracking),
        ), patch(
            "app.video_recognition.get_yolo_arcface_engine",
            return_value=engine,
        ), patch(
            "app.video_recognition.lock_anonymous_matching"
        ), patch(
            "app.video_recognition.find_closest_face", side_effect=matches
        ):
            result = recognize_video_tracks(
                MagicMock(),
                "videos/test/source.mp4",
                3.0,
            )

        self.assertEqual(recognizer.batch_sizes, [3])
        self.assertEqual(len(result.tracks), 1)
        self.assertEqual(result.tracks[0].face_id, face_id)
        self.assertEqual(result.tracks[0].status, "known")
        self.assertAlmostEqual(result.tracks[0].similarity, 0.81)

    def test_single_observation_unknown_does_not_create_anonymous_identity(self):
        frames = [
            _tracked_frame(
                0,
                0,
                (TrackedVideoFace(1, True, _detection(0, (10, 10, 40, 50))),),
            )
        ]
        tracking = _tracking_summary(1, 1, observation_count=1)
        recognizer = _FakeRecognizer(((1.0, 0.0),))
        engine = SimpleNamespace(recognizer=recognizer)

        with patch(
            "app.video_recognition.track_video_faces",
            side_effect=self._run_tracking(frames, tracking),
        ), patch(
            "app.video_recognition.get_yolo_arcface_engine",
            return_value=engine,
        ), patch(
            "app.video_recognition.lock_anonymous_matching"
        ), patch(
            "app.video_recognition.find_closest_face", return_value=None
        ), patch(
            "app.video_recognition.create_anonymous_identity"
        ) as create_identity:
            result = recognize_video_tracks(
                MagicMock(),
                "videos/test/source.mp4",
                3.0,
            )

        self.assertEqual(result.tracks, ())
        create_identity.assert_not_called()

    def test_splits_one_spatial_track_when_identity_changes_between_windows(self):
        known_face_id = UUID("77777777-7777-4777-8777-777777777777")
        anonymous_face_id = UUID("88888888-8888-4888-8888-888888888888")
        person = SimpleNamespace(
            id=7,
            face_id=known_face_id,
            first_name="Kenan",
            last_name="Imirzalioglu",
            description=None,
        )
        frames = [
            _tracked_frame(
                index,
                timestamp_ms,
                (TrackedVideoFace(1, index == 0, _detection(0, (10, 10, 50, 60))),),
            )
            for index, timestamp_ms in enumerate((0, 300, 600, 3300, 3600, 3900))
        ]
        tracking = VideoTrackingSummary(
            detection=_tracking_summary(1, 6).detection,
            unique_track_count=1,
            tracks=(
                VideoTrackSummary(
                    track_id=1,
                    first_seen_ms=0,
                    last_seen_ms=3900,
                    first_frame_number=0,
                    last_frame_number=5,
                    observation_count=6,
                    best_confidence=0.9,
                    best_frame_number=0,
                ),
            ),
        )
        recognizer = _FakeRecognizer(
            (
                (1.0, 0.0),
                (0.99, 0.01),
                (0.98, 0.02),
                (0.0, 1.0),
                (0.01, 0.99),
                (0.02, 0.98),
            )
        )
        engine = SimpleNamespace(recognizer=recognizer)
        identity = SimpleNamespace(face_id=anonymous_face_id, observation_count=1)
        sample = SimpleNamespace(image_path=None)
        known_match = (
            person,
            SimpleNamespace(image_path="persons/7/kenan.jpg"),
            0.64,
        )

        with patch(
            "app.video_recognition.track_video_faces",
            side_effect=self._run_tracking(frames, tracking),
        ), patch(
            "app.video_recognition.get_video_recognition_settings",
            return_value=_legacy_anonymous_settings(),
        ), patch(
            "app.video_recognition.get_yolo_arcface_engine",
            return_value=engine,
        ), patch(
            "app.video_recognition.lock_anonymous_matching"
        ), patch(
            "app.video_recognition.find_closest_face",
            side_effect=[None, None, None, known_match, known_match, known_match],
        ), patch(
            "app.video_recognition.find_closest_anonymous_face", return_value=None
        ), patch(
            "app.video_recognition.create_anonymous_identity",
            return_value=(identity, sample),
        ), patch(
            "app.video_recognition.save_anonymous_face_crop",
            return_value=f"anonymous/{anonymous_face_id}/sample.jpg",
        ):
            result = recognize_video_tracks(
                MagicMock(),
                "videos/test/source.mp4",
                3.0,
                manage_transaction=False,
            )

        self.assertEqual(recognizer.batch_sizes, [6])
        self.assertEqual([track.status for track in result.tracks], ["new_anonymous", "known"])
        self.assertEqual([track.source_track_id for track in result.tracks], [1, 1])
        self.assertEqual([track.track_id for track in result.tracks], [1, 2])
        self.assertEqual(result.tracks[0].last_seen_ms, 600)
        self.assertEqual(result.tracks[1].first_seen_ms, 3300)
        self.assertEqual(result.tracks[1].face_id, known_face_id)

    def test_bridges_one_uncertain_window_between_same_known_identity(self):
        face_id = UUID("66666666-6666-4666-8666-666666666666")
        person = SimpleNamespace(
            id=6,
            face_id=face_id,
            first_name="Ayni",
            last_name="Kisi",
            description=None,
        )
        timestamps = (0, 300, 3300, 3600, 6300, 6600)
        frames = [
            _tracked_frame(
                index,
                timestamp_ms,
                (TrackedVideoFace(1, index == 0, _detection(0, (10, 10, 50, 60))),),
            )
            for index, timestamp_ms in enumerate(timestamps)
        ]
        tracking = _tracking_summary(1, 6, observation_count=6)
        recognizer = _FakeRecognizer(((1.0, 0.0),) * 6)
        engine = SimpleNamespace(recognizer=recognizer)
        known_match = (
            person,
            SimpleNamespace(image_path="persons/6/a.jpg"),
            0.72,
        )

        with patch(
            "app.video_recognition.track_video_faces",
            side_effect=self._run_tracking(frames, tracking),
        ), patch(
            "app.video_recognition.get_yolo_arcface_engine",
            return_value=engine,
        ), patch(
            "app.video_recognition.lock_anonymous_matching"
        ), patch(
            "app.video_recognition.find_closest_face",
            side_effect=[known_match, known_match, None, None, known_match, known_match],
        ):
            result = recognize_video_tracks(
                MagicMock(),
                "videos/test/source.mp4",
                3.0,
            )

        self.assertEqual(len(result.tracks), 1)
        self.assertEqual(result.tracks[0].face_id, face_id)
        self.assertEqual(result.tracks[0].first_seen_ms, 0)
        self.assertEqual(result.tracks[0].last_seen_ms, 6600)

    def test_keeps_known_identity_through_later_side_profile_windows(self):
        face_id = UUID("12121212-1212-4212-8212-121212121212")
        person = SimpleNamespace(
            id=12,
            face_id=face_id,
            first_name="Kivanc",
            last_name="Tatlitug",
            description=None,
        )
        timestamps = (
            0, 300,
            3300, 3600,
            6300, 6600,
            9300, 9600,
            12300, 12600,
        )
        frames = [
            _tracked_frame(
                index,
                timestamp_ms,
                (TrackedVideoFace(1, index == 0, _detection(0, (10, 10, 50, 60))),),
            )
            for index, timestamp_ms in enumerate(timestamps)
        ]
        tracking = _tracking_summary(1, 10, observation_count=10)
        recognizer = _FakeRecognizer(
            ((1.0, 0.0),) * 2
            + ((0.4, 0.916515),) * 8
        )
        engine = SimpleNamespace(recognizer=recognizer)
        known_match = (
            person,
            SimpleNamespace(image_path="persons/12/kivanc.jpg"),
            0.72,
        )

        with patch(
            "app.video_recognition.track_video_faces",
            side_effect=self._run_tracking(frames, tracking),
        ), patch(
            "app.video_recognition.get_yolo_arcface_engine",
            return_value=engine,
        ), patch(
            "app.video_recognition.lock_anonymous_matching"
        ), patch(
            "app.video_recognition.find_closest_face",
            side_effect=[known_match, known_match] + [None] * 8,
        ), patch(
            "app.video_recognition.create_anonymous_identity"
        ) as create_identity:
            result = recognize_video_tracks(
                MagicMock(),
                "videos/test/source.mp4",
                3.0,
            )

        self.assertEqual(len(result.tracks), 1)
        self.assertEqual(result.tracks[0].status, "known")
        self.assertEqual(result.tracks[0].face_id, face_id)
        self.assertEqual(result.tracks[0].last_seen_ms, 12600)
        create_identity.assert_not_called()

    def test_does_not_extend_distant_known_identity_below_adaptive_threshold(self):
        face_id = UUID("13131313-1313-4313-8313-131313131313")
        person = SimpleNamespace(
            id=13,
            face_id=face_id,
            first_name="Guvenli",
            last_name="Esik",
            description=None,
        )
        timestamps = (
            0, 300,
            3300, 3600,
            6300, 6600,
            9300, 9600,
            12300, 12600,
        )
        frames = [
            _tracked_frame(
                index,
                timestamp_ms,
                (TrackedVideoFace(1, index == 0, _detection(0, (10, 10, 50, 60))),),
            )
            for index, timestamp_ms in enumerate(timestamps)
        ]
        tracking = _tracking_summary(1, 10, observation_count=10)
        recognizer = _FakeRecognizer(
            ((1.0, 0.0),) * 2
            + ((0.38, 0.924986),) * 8
        )
        known_match = (
            person,
            SimpleNamespace(image_path="persons/13/reference.jpg"),
            0.72,
        )

        with patch(
            "app.video_recognition.track_video_faces",
            side_effect=self._run_tracking(frames, tracking),
        ), patch(
            "app.video_recognition.get_yolo_arcface_engine",
            return_value=SimpleNamespace(recognizer=recognizer),
        ), patch(
            "app.video_recognition.lock_anonymous_matching"
        ), patch(
            "app.video_recognition.find_closest_face",
            side_effect=[known_match, known_match] + [None] * 8,
        ), patch(
            "app.video_recognition.create_anonymous_identity"
        ) as create_identity:
            result = recognize_video_tracks(
                MagicMock(),
                "videos/test/source.mp4",
                3.0,
            )

        self.assertEqual(len(result.tracks), 1)
        self.assertEqual(result.tracks[0].face_id, face_id)
        self.assertEqual(result.tracks[0].last_seen_ms, 9600)
        create_identity.assert_not_called()

    def test_does_not_persist_low_pose_quality_as_anonymous(self):
        frames = [
            _tracked_frame(
                index,
                index * 300,
                (TrackedVideoFace(1, index == 0, _detection(0, (10, 10, 50, 60))),),
            )
            for index in range(2)
        ]
        tracking = _tracking_summary(1, 2, observation_count=2)
        recognizer = _FakeRecognizer(((1.0, 0.0), (0.99, 0.01)))
        engine = SimpleNamespace(recognizer=recognizer)

        with patch(
            "app.video_recognition.track_video_faces",
            side_effect=self._run_tracking(frames, tracking),
        ), patch(
            "app.video_recognition.get_yolo_arcface_engine",
            return_value=engine,
        ), patch(
            "app.video_recognition._frontal_score",
            return_value=0.1,
        ), patch(
            "app.video_recognition.lock_anonymous_matching"
        ), patch(
            "app.video_recognition.find_closest_face", return_value=None
        ), patch(
            "app.video_recognition.create_anonymous_identity"
        ) as create_identity:
            result = recognize_video_tracks(
                MagicMock(),
                "videos/test/source.mp4",
                3.0,
            )

        self.assertEqual(result.tracks, ())
        create_identity.assert_not_called()

    def test_suppresses_a_short_weak_known_match(self):
        face_id = UUID("34343434-3434-4434-8434-343434343434")
        person = SimpleNamespace(
            id=34,
            face_id=face_id,
            first_name="Yanlis",
            last_name="Eslesme",
            description=None,
        )
        frames = [
            _tracked_frame(
                index,
                index * 300,
                (TrackedVideoFace(1, index == 0, _detection(0, (10, 10, 50, 60))),),
            )
            for index in range(3)
        ]
        tracking = _tracking_summary(1, 3, observation_count=3)
        recognizer = _FakeRecognizer(((1.0, 0.0),) * 3)
        engine = SimpleNamespace(recognizer=recognizer)
        weak_match = (
            person,
            SimpleNamespace(image_path="persons/34/reference.jpg"),
            0.4752,
        )

        with patch(
            "app.video_recognition.track_video_faces",
            side_effect=self._run_tracking(frames, tracking),
        ), patch(
            "app.video_recognition.get_yolo_arcface_engine",
            return_value=engine,
        ), patch(
            "app.video_recognition.lock_anonymous_matching"
        ), patch(
            "app.video_recognition.find_closest_face",
            side_effect=[weak_match, weak_match, weak_match],
        ):
            result = recognize_video_tracks(
                MagicMock(),
                "videos/test/source.mp4",
                3.0,
            )

        self.assertEqual(result.tracks, ())

    def test_retains_a_persistent_weak_known_match(self):
        face_id = UUID("56565656-5656-4656-8656-565656565656")
        person = SimpleNamespace(
            id=56,
            face_id=face_id,
            first_name="Kararli",
            last_name="Eslesme",
            description=None,
        )
        frames = [
            _tracked_frame(
                index,
                index * 300,
                (TrackedVideoFace(1, index == 0, _detection(0, (10, 10, 50, 60))),),
            )
            for index in range(5)
        ]
        tracking = _tracking_summary(1, 5, observation_count=5)
        recognizer = _FakeRecognizer(((1.0, 0.0),) * 3)
        engine = SimpleNamespace(recognizer=recognizer)
        weak_match = (
            person,
            SimpleNamespace(image_path="persons/56/reference.jpg"),
            0.4752,
        )

        with patch(
            "app.video_recognition.track_video_faces",
            side_effect=self._run_tracking(frames, tracking),
        ), patch(
            "app.video_recognition.get_yolo_arcface_engine",
            return_value=engine,
        ), patch(
            "app.video_recognition.lock_anonymous_matching"
        ), patch(
            "app.video_recognition.find_closest_face",
            side_effect=[weak_match, weak_match, weak_match],
        ):
            result = recognize_video_tracks(
                MagicMock(),
                "videos/test/source.mp4",
                3.0,
            )

        self.assertEqual(len(result.tracks), 1)
        self.assertEqual(result.tracks[0].status, "known")
        self.assertEqual(result.tracks[0].face_id, face_id)
        self.assertEqual(result.tracks[0].last_seen_ms, 1200)

    def test_suppresses_a_short_strong_match_with_low_detection_confidence(self):
        face_id = UUID("67676767-6767-4767-8767-676767676767")
        person = SimpleNamespace(
            id=67,
            face_id=face_id,
            first_name="Dusuk",
            last_name="Kalite",
            description=None,
        )
        frames = [
            _tracked_frame(
                index,
                index * 300,
                (
                    TrackedVideoFace(
                        1,
                        index == 0,
                        _detection(0, (10, 10, 50, 60), confidence=0.30),
                    ),
                ),
            )
            for index in range(3)
        ]
        tracking = _tracking_summary(1, 3, observation_count=3)
        recognizer = _FakeRecognizer(((1.0, 0.0),) * 3)
        engine = SimpleNamespace(recognizer=recognizer)
        strong_match = (
            person,
            SimpleNamespace(image_path="persons/67/reference.jpg"),
            0.80,
        )

        with patch(
            "app.video_recognition.track_video_faces",
            side_effect=self._run_tracking(frames, tracking),
        ), patch(
            "app.video_recognition.get_yolo_arcface_engine",
            return_value=engine,
        ), patch(
            "app.video_recognition.lock_anonymous_matching"
        ), patch(
            "app.video_recognition.find_closest_face",
            side_effect=[strong_match, strong_match, strong_match],
        ):
            result = recognize_video_tracks(
                MagicMock(),
                "videos/test/source.mp4",
                3.0,
            )

        self.assertEqual(result.tracks, ())

    def test_reuses_new_anonymous_face_id_for_a_split_track(self):
        frames = [
            _tracked_frame(
                0,
                0,
                (TrackedVideoFace(1, True, _detection(0, (10, 10, 40, 50))),),
            ),
            _tracked_frame(
                1,
                300,
                (TrackedVideoFace(1, False, _detection(0, (11, 10, 41, 50))),),
            ),
            _tracked_frame(
                10,
                2000,
                (TrackedVideoFace(2, True, _detection(0, (12, 10, 42, 50))),),
            ),
            _tracked_frame(
                11,
                2300,
                (TrackedVideoFace(2, False, _detection(0, (13, 10, 43, 50))),),
            ),
        ]
        tracking = _tracking_summary(2, 4)
        embedding = np.asarray([1.0, 0.0], dtype=np.float32)
        returned_embedding = np.asarray([0.8, 0.6], dtype=np.float32)
        recognizer = _FakeRecognizer(
            (embedding, embedding, returned_embedding, returned_embedding)
        )
        engine = SimpleNamespace(recognizer=recognizer)
        face_id = UUID("33333333-3333-3333-3333-333333333333")
        identity = SimpleNamespace(face_id=face_id, observation_count=1)
        duplicate_identity = SimpleNamespace(
            face_id=UUID("99999999-9999-4999-8999-999999999998"),
            observation_count=1,
        )
        sample = SimpleNamespace(image_path=None)
        session = MagicMock()

        with patch(
            "app.video_recognition.track_video_faces",
            side_effect=self._run_tracking(frames, tracking),
        ), patch(
            "app.video_recognition.get_video_recognition_settings",
            return_value=_legacy_anonymous_settings(),
        ), patch(
            "app.video_recognition.get_yolo_arcface_engine",
            return_value=engine,
        ), patch(
            "app.video_recognition.lock_anonymous_matching"
        ), patch(
            "app.video_recognition.find_closest_face", return_value=None
        ), patch(
            "app.video_recognition.find_closest_anonymous_face",
            side_effect=[None, (duplicate_identity, 0.99)],
        ), patch(
            "app.video_recognition.create_anonymous_identity",
            return_value=(identity, sample),
        ) as create_identity, patch(
            "app.video_recognition.save_anonymous_face_crop",
            return_value=f"anonymous/{face_id}/sample.jpg",
        ), patch(
            "app.video_recognition.synchronize_face_id_safely"
        ) as synchronize:
            result = recognize_video_tracks(
                session,
                "videos/test/source.mp4",
                3.0,
            )

        self.assertEqual([track.status for track in result.tracks], ["new_anonymous", "anonymous"])
        self.assertEqual({track.face_id for track in result.tracks}, {face_id})
        self.assertTrue(all(track.name is None for track in result.tracks))
        self.assertTrue(all(track.metadata is None for track in result.tracks))
        self.assertEqual(identity.observation_count, 2)
        create_identity.assert_called_once()
        session.commit.assert_called_once()
        synchronize.assert_called_once_with(session, face_id)

    def test_can_defer_anonymous_commit_to_video_transaction(self):
        frames = [
            _tracked_frame(
                0,
                0,
                (TrackedVideoFace(1, True, _detection(0, (10, 10, 40, 50))),),
            ),
            _tracked_frame(
                1,
                300,
                (TrackedVideoFace(1, False, _detection(0, (11, 10, 41, 50))),),
            ),
        ]
        tracking = _tracking_summary(1, 2)
        recognizer = _FakeRecognizer(((1.0, 0.0), (0.99, 0.01)))
        engine = SimpleNamespace(recognizer=recognizer)
        face_id = UUID("44444444-4444-4444-4444-444444444444")
        identity = SimpleNamespace(face_id=face_id, observation_count=1)
        sample = SimpleNamespace(image_path=None)
        session = MagicMock()

        with patch(
            "app.video_recognition.track_video_faces",
            side_effect=self._run_tracking(frames, tracking),
        ), patch(
            "app.video_recognition.get_video_recognition_settings",
            return_value=_legacy_anonymous_settings(),
        ), patch(
            "app.video_recognition.get_yolo_arcface_engine",
            return_value=engine,
        ), patch(
            "app.video_recognition.lock_anonymous_matching"
        ), patch(
            "app.video_recognition.find_closest_face", return_value=None
        ), patch(
            "app.video_recognition.find_closest_anonymous_face", return_value=None
        ), patch(
            "app.video_recognition.create_anonymous_identity",
            return_value=(identity, sample),
        ), patch(
            "app.video_recognition.save_anonymous_face_crop",
            return_value=f"anonymous/{face_id}/sample.jpg",
        ), patch(
            "app.video_recognition.synchronize_face_id_safely"
        ) as synchronize:
            result = recognize_video_tracks(
                session,
                "videos/test/source.mp4",
                3.0,
                manage_transaction=False,
            )

        session.commit.assert_not_called()
        synchronize.assert_not_called()
        self.assertEqual(result.changed_face_ids, (face_id,))
        self.assertEqual(
            result.stored_anonymous_image_paths,
            (f"anonymous/{face_id}/sample.jpg",),
        )

    def test_read_only_mode_keeps_new_anonymous_identity_in_memory(self):
        frames = [
            _tracked_frame(
                0,
                0,
                (TrackedVideoFace(1, True, _detection(0, (10, 10, 40, 50))),),
            ),
            _tracked_frame(
                1,
                300,
                (TrackedVideoFace(1, False, _detection(0, (11, 10, 41, 50))),),
            ),
        ]
        tracking = _tracking_summary(1, 2)
        recognizer = _FakeRecognizer(((1.0, 0.0), (0.99, 0.01)))
        engine = SimpleNamespace(recognizer=recognizer)
        session = MagicMock()

        with patch(
            "app.video_recognition.track_video_faces",
            side_effect=self._run_tracking(frames, tracking),
        ), patch(
            "app.video_recognition.get_video_recognition_settings",
            return_value=_legacy_anonymous_settings(),
        ), patch(
            "app.video_recognition.get_yolo_arcface_engine",
            return_value=engine,
        ), patch(
            "app.video_recognition.find_closest_face", return_value=None
        ), patch(
            "app.video_recognition.find_closest_anonymous_face", return_value=None
        ), patch(
            "app.video_recognition.lock_anonymous_matching"
        ) as lock_anonymous, patch(
            "app.video_recognition.create_anonymous_identity"
        ) as create_identity, patch(
            "app.video_recognition.record_anonymous_observation"
        ) as record_observation, patch(
            "app.video_recognition.save_anonymous_face_crop"
        ) as save_crop:
            result = recognize_video_tracks(
                session,
                "videos/test/source.mp4",
                3.0,
                manage_transaction=False,
                read_only=True,
            )

        self.assertEqual(len(result.tracks), 1)
        self.assertEqual(result.tracks[0].status, "new_anonymous")
        self.assertEqual(result.changed_face_ids, ())
        self.assertEqual(result.stored_anonymous_image_paths, ())
        lock_anonymous.assert_not_called()
        create_identity.assert_not_called()
        record_observation.assert_not_called()
        save_crop.assert_not_called()
        session.commit.assert_not_called()

    def test_accepts_a_stable_quality_anonymous_track(self):
        frames = [
            _textured_tracked_frame(
                index,
                index * 300,
                (TrackedVideoFace(1, index == 0, _detection(0, (10, 10, 90, 90))),),
            )
            for index in range(4)
        ]
        tracking = _tracking_summary(1, 4, observation_count=4)
        recognizer = _FakeRecognizer(((1.0, 0.0),) * 3)

        with patch(
            "app.video_recognition.track_video_faces",
            side_effect=self._run_tracking(frames, tracking),
        ), patch(
            "app.video_recognition.get_yolo_arcface_engine",
            return_value=SimpleNamespace(recognizer=recognizer),
        ), patch(
            "app.video_recognition.find_closest_face", return_value=None
        ), patch(
            "app.video_recognition.find_closest_anonymous_face", return_value=None
        ):
            result = recognize_video_tracks(
                MagicMock(),
                "videos/test/source.mp4",
                3.0,
                manage_transaction=False,
                read_only=True,
            )

        self.assertEqual(len(result.tracks), 1)
        self.assertEqual(result.tracks[0].status, "new_anonymous")

    def test_accepts_one_exceptionally_clear_anonymous_sample(self):
        frames = [
            _textured_tracked_frame(
                0,
                0,
                (TrackedVideoFace(1, True, _detection(0, (10, 10, 90, 90))),),
            )
        ]
        tracking = _tracking_summary(1, 1, observation_count=1)
        recognizer = _FakeRecognizer(((1.0, 0.0),))

        with patch(
            "app.video_recognition.track_video_faces",
            side_effect=self._run_tracking(frames, tracking),
        ), patch(
            "app.video_recognition.get_yolo_arcface_engine",
            return_value=SimpleNamespace(recognizer=recognizer),
        ), patch(
            "app.video_recognition._frontal_score", return_value=0.9
        ), patch(
            "app.video_recognition.find_closest_face", return_value=None
        ), patch(
            "app.video_recognition.find_closest_anonymous_face", return_value=None
        ):
            result = recognize_video_tracks(
                MagicMock(),
                "videos/test/source.mp4",
                3.0,
                manage_transaction=False,
                read_only=True,
            )

        self.assertEqual(len(result.tracks), 1)
        self.assertEqual(result.tracks[0].status, "new_anonymous")
        self.assertEqual(result.tracks[0].observation_count, 1)

    def test_accepts_a_short_consistent_high_quality_anonymous_track(self):
        frames = [
            _textured_tracked_frame(
                index,
                index * 200,
                (
                    TrackedVideoFace(
                        1,
                        index == 0,
                        _detection(0, (10, 10, 90, 90)),
                    ),
                ),
            )
            for index in range(4)
        ]
        tracking = _tracking_summary(1, 4, observation_count=4)
        recognizer = _FakeRecognizer(
            ((1.0, 0.0), (0.99, 0.01), (0.98, 0.02))
        )

        with patch(
            "app.video_recognition.track_video_faces",
            side_effect=self._run_tracking(frames, tracking),
        ), patch(
            "app.video_recognition.get_yolo_arcface_engine",
            return_value=SimpleNamespace(recognizer=recognizer),
        ), patch(
            "app.video_recognition._frontal_score", return_value=0.9
        ), patch(
            "app.video_recognition.find_closest_face", return_value=None
        ), patch(
            "app.video_recognition.find_closest_anonymous_face", return_value=None
        ):
            result = recognize_video_tracks(
                MagicMock(),
                "videos/test/source.mp4",
                3.0,
                manage_transaction=False,
                read_only=True,
            )

        self.assertEqual(len(result.tracks), 1)
        self.assertEqual(result.tracks[0].status, "new_anonymous")
        self.assertEqual(result.tracks[0].observation_count, 4)

    def test_rejects_a_three_frame_scene_transition(self):
        frames = [
            _textured_tracked_frame(
                index,
                index * 160,
                (
                    TrackedVideoFace(
                        1,
                        index == 0,
                        _detection(0, (10, 10, 90, 90)),
                    ),
                ),
            )
            for index in range(3)
        ]
        tracking = _tracking_summary(1, 3, observation_count=3)
        recognizer = _FakeRecognizer(
            ((1.0, 0.0), (0.99, 0.01), (0.98, 0.02))
        )

        with patch(
            "app.video_recognition.track_video_faces",
            side_effect=self._run_tracking(frames, tracking),
        ), patch(
            "app.video_recognition.get_yolo_arcface_engine",
            return_value=SimpleNamespace(recognizer=recognizer),
        ), patch(
            "app.video_recognition._frontal_score", return_value=0.9
        ), patch(
            "app.video_recognition.find_closest_face", return_value=None
        ):
            result = recognize_video_tracks(
                MagicMock(),
                "videos/test/source.mp4",
                3.0,
                manage_transaction=False,
                read_only=True,
            )

        self.assertEqual(result.tracks, ())

    def test_rejects_a_short_high_quality_track_without_embedding_majority(self):
        frames = [
            _textured_tracked_frame(
                index,
                index * 150,
                (
                    TrackedVideoFace(
                        1,
                        index == 0,
                        _detection(0, (10, 10, 90, 90)),
                    ),
                ),
            )
            for index in range(3)
        ]
        tracking = _tracking_summary(1, 3, observation_count=3)
        recognizer = _FakeRecognizer(
            ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0))
        )

        with patch(
            "app.video_recognition.track_video_faces",
            side_effect=self._run_tracking(frames, tracking),
        ), patch(
            "app.video_recognition.get_yolo_arcface_engine",
            return_value=SimpleNamespace(recognizer=recognizer),
        ), patch(
            "app.video_recognition._frontal_score", return_value=0.9
        ), patch(
            "app.video_recognition.find_closest_face", return_value=None
        ):
            result = recognize_video_tracks(
                MagicMock(),
                "videos/test/source.mp4",
                3.0,
                manage_transaction=False,
                read_only=True,
            )

        self.assertEqual(result.tracks, ())

    def test_rejects_one_merely_moderate_anonymous_sample(self):
        frames = [
            _textured_tracked_frame(
                0,
                0,
                (
                    TrackedVideoFace(
                        1,
                        True,
                        _detection(0, (10, 10, 90, 90), confidence=0.70),
                    ),
                ),
            )
        ]
        tracking = _tracking_summary(1, 1, observation_count=1)
        recognizer = _FakeRecognizer(((1.0, 0.0),))

        with patch(
            "app.video_recognition.track_video_faces",
            side_effect=self._run_tracking(frames, tracking),
        ), patch(
            "app.video_recognition.get_yolo_arcface_engine",
            return_value=SimpleNamespace(recognizer=recognizer),
        ), patch(
            "app.video_recognition._frontal_score", return_value=0.9
        ), patch(
            "app.video_recognition.find_closest_face", return_value=None
        ):
            result = recognize_video_tracks(
                MagicMock(),
                "videos/test/source.mp4",
                3.0,
                manage_transaction=False,
                read_only=True,
            )

        self.assertEqual(result.tracks, ())

    def test_rejects_a_stable_but_too_small_anonymous_track(self):
        frames = [
            _textured_tracked_frame(
                index,
                index * 300,
                (TrackedVideoFace(1, index == 0, _detection(0, (10, 10, 45, 55))),),
            )
            for index in range(4)
        ]
        tracking = _tracking_summary(1, 4, observation_count=4)
        recognizer = _FakeRecognizer(((1.0, 0.0),) * 3)

        with patch(
            "app.video_recognition.track_video_faces",
            side_effect=self._run_tracking(frames, tracking),
        ), patch(
            "app.video_recognition.get_yolo_arcface_engine",
            return_value=SimpleNamespace(recognizer=recognizer),
        ), patch(
            "app.video_recognition.find_closest_face", return_value=None
        ):
            result = recognize_video_tracks(
                MagicMock(),
                "videos/test/source.mp4",
                3.0,
                manage_transaction=False,
                read_only=True,
            )

        self.assertEqual(result.tracks, ())

    def test_rejects_anonymous_samples_with_inconsistent_embeddings(self):
        frames = [
            _textured_tracked_frame(
                index,
                index * 300,
                (TrackedVideoFace(1, index == 0, _detection(0, (10, 10, 90, 90))),),
            )
            for index in range(4)
        ]
        tracking = _tracking_summary(1, 4, observation_count=4)
        recognizer = _FakeRecognizer(((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)))

        with patch(
            "app.video_recognition.track_video_faces",
            side_effect=self._run_tracking(frames, tracking),
        ), patch(
            "app.video_recognition.get_yolo_arcface_engine",
            return_value=SimpleNamespace(recognizer=recognizer),
        ), patch(
            "app.video_recognition.find_closest_face", return_value=None
        ):
            result = recognize_video_tracks(
                MagicMock(),
                "videos/test/source.mp4",
                3.0,
                manage_transaction=False,
                read_only=True,
            )

        self.assertEqual(result.tracks, ())

    def test_returns_empty_result_without_running_arcface_when_no_face_exists(self):
        tracking = _tracking_summary(0, 0)
        recognizer = _FakeRecognizer(())
        engine = SimpleNamespace(recognizer=recognizer)
        session = MagicMock()

        with patch(
            "app.video_recognition.track_video_faces",
            side_effect=self._run_tracking([], tracking),
        ), patch(
            "app.video_recognition.get_yolo_arcface_engine",
            return_value=engine,
        ):
            result = recognize_video_tracks(
                session,
                "videos/test/source.mp4",
                3.0,
            )

        self.assertEqual(result.tracks, ())
        self.assertEqual(recognizer.batch_sizes, [])
        session.commit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
