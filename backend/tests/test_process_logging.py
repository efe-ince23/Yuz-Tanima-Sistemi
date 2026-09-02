import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from uuid import uuid4

from app.process_tracking import (
    begin_process,
    complete_process,
    fail_process,
    read_fallback_process,
)


class ProcessLoggingResilienceTests(unittest.TestCase):
    @patch("app.process_tracking._write_fallback_log", return_value=True)
    @patch("app.process_tracking.SessionLocal", side_effect=RuntimeError("database unavailable"))
    def test_database_log_failure_never_raises(self, _session, fallback_log):
        process_id = uuid4()

        self.assertFalse(begin_process(process_id, "detect"))
        self.assertFalse(
            complete_process(
                process_id,
                operation_type="detect",
                status="no_face",
                http_status=200,
                face_count=0,
                faces=[],
                result={"status": "no_face"},
            )
        )
        self.assertFalse(fail_process(process_id, "detect", 500, "log error"))
        self.assertEqual(fallback_log.call_count, 3)

    def test_fallback_log_is_persistent_and_queryable(self):
        process_id = uuid4()

        with TemporaryDirectory() as temp_dir:
            log_root = Path(temp_dir)
            log_path = log_root / "recognition-processes.jsonl"
            with patch("app.process_tracking.PROCESS_LOG_ROOT", log_root), patch(
                "app.process_tracking.PROCESS_FALLBACK_LOG", log_path
            ), patch(
                "app.process_tracking.SessionLocal",
                side_effect=RuntimeError("database unavailable"),
            ):
                self.assertFalse(begin_process(process_id, "identify"))
                self.assertFalse(
                    complete_process(
                        process_id,
                        operation_type="identify",
                        status="completed",
                        http_status=200,
                        face_count=1,
                        faces=[{"face_id": "face-123", "status": "known"}],
                        result={"recognized": True},
                    )
                )

                stored = read_fallback_process(process_id)

        self.assertIsNotNone(stored)
        self.assertEqual(stored["process_id"], str(process_id))
        self.assertEqual(stored["task_detail"]["operation_type"], "identify")
        self.assertEqual(stored["task_detail"]["processed_face_count"], 1)
        self.assertEqual(
            stored["task_detail"]["faces"],
            [{"face_id": "face-123", "status": "known"}],
        )


if __name__ == "__main__":
    unittest.main()
