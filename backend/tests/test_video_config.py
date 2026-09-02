import os
import unittest
from unittest.mock import patch

from app.video_config import (
    get_video_recognition_settings,
    get_video_settings,
    get_video_tracking_settings,
)


class VideoSettingsTests(unittest.TestCase):
    def tearDown(self) -> None:
        get_video_settings.cache_clear()
        get_video_tracking_settings.cache_clear()
        get_video_recognition_settings.cache_clear()

    def test_loads_safe_video_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            get_video_settings.cache_clear()
            settings = get_video_settings()

        self.assertEqual(settings.max_size_bytes, 200 * 1024 * 1024)
        self.assertEqual(settings.max_duration_seconds, 300)
        self.assertEqual(settings.sample_fps, 6)
        self.assertEqual(settings.processing_concurrency, 1)
        self.assertEqual(settings.arcface_frame_batch_size, 1)
        self.assertTrue(settings.supports_content_type("video/mp4; charset=binary"))
        self.assertTrue(settings.supports_container("MP4"))
        self.assertTrue(settings.supports_codec("H264"))
        self.assertFalse(settings.supports_codec("hevc"))

    def test_environment_values_are_normalized(self):
        environment = {
            "VIDEO_MAX_SIZE_MB": "150",
            "VIDEO_MAX_DURATION_SECONDS": "120.5",
            "VIDEO_SAMPLE_FPS": "5",
            "VIDEO_PROCESSING_CONCURRENCY": "2",
            "VIDEO_ARCFACE_FRAME_BATCH_SIZE": "6",
            "VIDEO_ALLOWED_CONTENT_TYPES": " video/mp4, video/webm,video/mp4 ",
            "VIDEO_ALLOWED_CONTAINERS": "mp4,webm",
            "VIDEO_ALLOWED_CODECS": "h264,vp9",
        }
        with patch.dict(os.environ, environment, clear=True):
            get_video_settings.cache_clear()
            settings = get_video_settings()

        self.assertEqual(settings.max_size_bytes, 150 * 1024 * 1024)
        self.assertEqual(settings.max_duration_seconds, 120.5)
        self.assertEqual(settings.sample_fps, 5)
        self.assertEqual(settings.processing_concurrency, 2)
        self.assertEqual(settings.arcface_frame_batch_size, 6)
        self.assertEqual(settings.allowed_content_types, ("video/mp4", "video/webm"))

    def test_rejects_invalid_limits(self):
        for name, value in (
            ("VIDEO_MAX_SIZE_MB", "0"),
            ("VIDEO_MAX_DURATION_SECONDS", "invalid"),
            ("VIDEO_SAMPLE_FPS", "-1"),
            ("VIDEO_PROCESSING_CONCURRENCY", "0"),
            ("VIDEO_ARCFACE_FRAME_BATCH_SIZE", "0"),
        ):
            with self.subTest(name=name), patch.dict(
                os.environ, {name: value}, clear=True
            ):
                get_video_settings.cache_clear()
                with self.assertRaises(RuntimeError):
                    get_video_settings()

    def test_loads_tracking_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            get_video_tracking_settings.cache_clear()
            settings = get_video_tracking_settings()

        self.assertEqual(settings.max_gap_ms, 750)
        self.assertEqual(settings.min_iou, 0.10)
        self.assertEqual(settings.max_center_distance, 0.25)
        self.assertEqual(settings.max_area_ratio, 3.0)
        self.assertEqual(settings.engine, "custom")
        self.assertEqual(settings.bytetrack_activation_threshold, 0.15)
        self.assertEqual(settings.bytetrack_matching_threshold, 0.90)
        self.assertEqual(settings.bytetrack_minimum_consecutive_frames, 1)
        self.assertTrue(settings.scene_cut_enabled)
        self.assertEqual(settings.scene_cut_histogram_correlation, 0.35)
        self.assertEqual(settings.scene_cut_pixel_difference, 0.10)

    def test_loads_recognition_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            get_video_recognition_settings.cache_clear()
            settings = get_video_recognition_settings()

        self.assertEqual(settings.samples_per_track, 3)
        self.assertEqual(settings.anonymous_min_observations, 2)
        self.assertEqual(settings.anonymous_min_quality_samples, 3)
        self.assertEqual(settings.anonymous_min_quality, 0.50)
        self.assertEqual(settings.anonymous_min_frontal_score, 0.35)
        self.assertEqual(settings.anonymous_min_duration_ms, 800)
        self.assertEqual(settings.anonymous_min_detection_confidence, 0.50)
        self.assertEqual(settings.anonymous_min_face_size_px, 64)
        self.assertEqual(settings.anonymous_min_sharpness, 35)
        self.assertEqual(settings.anonymous_min_embedding_consistency, 0.35)
        self.assertEqual(settings.anonymous_single_min_quality, 0.60)
        self.assertEqual(settings.anonymous_single_min_frontal_score, 0.65)
        self.assertEqual(settings.anonymous_single_min_detection_confidence, 0.75)
        self.assertEqual(settings.anonymous_single_min_face_size_px, 80)
        self.assertEqual(settings.anonymous_single_min_sharpness, 55)
        self.assertEqual(settings.anonymous_short_min_observations, 4)
        self.assertEqual(settings.anonymous_short_min_duration_ms, 500)
        self.assertEqual(settings.window_ms, 3000)
        self.assertEqual(settings.strong_match_margin, 0.08)
        self.assertEqual(settings.weak_known_min_duration_ms, 1000)
        self.assertEqual(settings.short_known_min_detection_confidence, 0.50)
        self.assertEqual(settings.bridge_window_count, 1)
        self.assertEqual(settings.continuity_window_count, 4)
        self.assertEqual(settings.continuity_threshold, 0.30)
        self.assertEqual(settings.continuity_distance_penalty, 0.03)
        self.assertEqual(settings.reid_gallery_size, 5)
        self.assertEqual(settings.reid_min_similarity, 0.30)

    def test_rejects_invalid_tracking_limits(self):
        for name, value in (
            ("VIDEO_TRACK_MAX_GAP_SECONDS", "0"),
            ("VIDEO_TRACK_MIN_IOU", "1.1"),
            ("VIDEO_TRACK_MAX_CENTER_DISTANCE", "0"),
            ("VIDEO_TRACK_MAX_AREA_RATIO", "invalid"),
        ):
            with self.subTest(name=name), patch.dict(
                os.environ, {name: value}, clear=True
            ):
                get_video_tracking_settings.cache_clear()
                with self.assertRaises(RuntimeError):
                    get_video_tracking_settings()


if __name__ == "__main__":
    unittest.main()
