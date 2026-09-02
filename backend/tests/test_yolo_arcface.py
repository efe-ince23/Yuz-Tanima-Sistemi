import unittest

import numpy as np

from app.yolo_arcface import _has_plausible_face_geometry


class YoloFaceGeometryTests(unittest.TestCase):
    def test_accepts_regular_and_small_face_boxes(self) -> None:
        self.assertTrue(
            _has_plausible_face_geometry(
                np.asarray([10.0, 20.0, 90.0, 130.0], dtype=np.float32)
            )
        )
        self.assertTrue(
            _has_plausible_face_geometry(
                np.asarray([5.0, 5.0, 23.0, 40.0], dtype=np.float32)
            )
        )

    def test_rejects_extreme_aspect_ratios(self) -> None:
        self.assertFalse(
            _has_plausible_face_geometry(
                np.asarray([0.0, 20.0, 18.0, 166.0], dtype=np.float32)
            )
        )
        self.assertFalse(
            _has_plausible_face_geometry(
                np.asarray([20.0, 0.0, 140.0, 18.0], dtype=np.float32)
            )
        )

    def test_rejects_empty_boxes(self) -> None:
        self.assertFalse(
            _has_plausible_face_geometry(
                np.asarray([20.0, 20.0, 20.0, 80.0], dtype=np.float32)
            )
        )


if __name__ == "__main__":
    unittest.main()
