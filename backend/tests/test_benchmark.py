import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from benchmark.dataset import collect_identity_splits, load_verification_pairs
from benchmark.metrics import best_threshold, roc_auc, threshold_curve
from benchmark.report import write_reports
from benchmark.run import _select_lfw_target_face


class BenchmarkTests(unittest.TestCase):
    def test_selects_the_center_lfw_subject_when_background_faces_exist(self) -> None:
        image = np.zeros((250, 250, 3), dtype=np.uint8)
        background_face = SimpleNamespace(
            bbox=np.asarray([0.0, 30.0, 120.0, 210.0], dtype=np.float32)
        )
        target_face = SimpleNamespace(
            bbox=np.asarray([75.0, 55.0, 180.0, 205.0], dtype=np.float32)
        )

        selected = _select_lfw_target_face(
            image,
            [background_face, target_face],
        )

        self.assertIs(selected, target_face)

    def test_parses_official_lfw_pair_rows_and_identity_splits(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for name, count in (("Person_One", 2), ("Person_Two", 1)):
                person = root / name
                person.mkdir()
                for index in range(1, count + 1):
                    (person / f"{name}_{index:04d}.jpg").touch()
            pairs_file = root / "pairs.csv"
            with pairs_file.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(("name", "imagenum1", "imagenum2", ""))
                writer.writerow(("Person_One", "1", "2", ""))
                writer.writerow(("Person_One", "1", "Person_Two", "1"))

            pairs = load_verification_pairs(pairs_file, root, 10, seed=1)
            identities, unknowns = collect_identity_splits(root, 10, 10, seed=1)

        self.assertEqual(len(pairs), 2)
        self.assertEqual({pair.same_person for pair in pairs}, {True, False})
        self.assertEqual([item.name for item in identities], ["Person_One"])
        self.assertEqual(len(unknowns), 1)

    def test_finds_a_threshold_and_perfect_auc_for_separated_scores(self):
        curve = threshold_curve((0.8, 0.9), (0.1, 0.2))
        selected = best_threshold(curve)

        self.assertEqual(selected.accuracy, 1.0)
        self.assertGreaterEqual(selected.threshold, 0.2)
        self.assertLessEqual(selected.threshold, 0.8)
        self.assertGreater(roc_auc((0.8, 0.9), (0.1, 0.2)), 0.99)

    def test_writes_json_csv_and_html_reports(self):
        metric = best_threshold(threshold_curve((0.8,), (0.1,)))
        result = {
            "benchmark_id": "test-run",
            "created_at": "2026-08-26T00:00:00+00:00",
            "detection": {
                "images_processed": 2,
                "single_face_rate": 1.0,
                "latency_ms": {"p95": 10.0},
            },
            "verification": {
                "configured": {"accuracy": 1.0},
                "recommended": {"threshold": 0.5, "accuracy": 1.0},
                "roc_auc": 1.0,
            },
            "identification": {
                "rank1_accuracy": 1.0,
                "unknown_rejection_rate": 1.0,
            },
            "performance": {"arcface_batches": []},
            "warnings": [],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = write_reports(
                Path(temporary_directory) / "report",
                result,
                (metric,),
            )
            self.assertTrue(all(Path(path).is_file() for path in paths.values()))


if __name__ == "__main__":
    unittest.main()
