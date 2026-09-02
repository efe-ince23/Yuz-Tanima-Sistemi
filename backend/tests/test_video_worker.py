import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import UUID

from app.video_worker import VideoWorkerPool


class VideoWorkerPoolTests(unittest.TestCase):
    def test_submit_deduplicates_a_process_while_it_is_pending(self):
        process_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
        pool = VideoWorkerPool()
        executor = MagicMock()
        pool._executor = executor

        self.assertTrue(pool.submit(process_id))
        self.assertFalse(pool.submit(process_id))
        executor.submit.assert_called_once_with(pool._run, process_id)

    def test_worker_releases_process_id_after_completion(self):
        process_id = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
        pool = VideoWorkerPool()
        pool._submitted.add(process_id)
        session = MagicMock()
        session_factory = MagicMock()
        session_factory.return_value.__enter__.return_value = session

        with patch("app.video_worker.SessionLocal", session_factory), patch(
            "app.video_worker.process_video_job"
        ) as process_video:
            pool._run(process_id)

        process_video.assert_called_once_with(session, process_id)
        self.assertNotIn(process_id, pool._submitted)

    def test_worker_releases_process_id_after_failure(self):
        process_id = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
        pool = VideoWorkerPool()
        pool._submitted.add(process_id)
        session_factory = MagicMock()
        session_factory.return_value.__enter__.side_effect = RuntimeError("test")

        with patch("app.video_worker.SessionLocal", session_factory):
            pool._run(process_id)

        self.assertNotIn(process_id, pool._submitted)

    def test_recovery_requeues_an_interrupted_job(self):
        queued_id = UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")
        interrupted_id = UUID("eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee")
        queued = SimpleNamespace(process_id=queued_id, status="queued")
        interrupted_process = SimpleNamespace(
            status="processing",
            http_status=None,
            error_detail="old error",
            task_detail={"operation_type": "video_recognize", "status": "processing"},
        )
        interrupted = SimpleNamespace(
            process_id=interrupted_id,
            status="processing",
            progress_percent=45.0,
            error_code="old",
            error_detail="old error",
            process=interrupted_process,
        )
        session = MagicMock()
        session.scalars.return_value.all.return_value = [queued, interrupted]
        session_factory = MagicMock()
        session_factory.return_value.__enter__.return_value = session
        pool = VideoWorkerPool()

        with patch("app.video_worker.SessionLocal", session_factory):
            recovered = pool._recover_pending_jobs()

        self.assertEqual(recovered, [queued_id, interrupted_id])
        self.assertEqual(interrupted.status, "queued")
        self.assertEqual(interrupted.progress_percent, 0.0)
        self.assertEqual(interrupted_process.status, "queued")
        self.assertEqual(interrupted_process.http_status, 202)
        self.assertEqual(interrupted_process.task_detail["status"], "queued")
        session.commit.assert_called_once()


if __name__ == "__main__":
    unittest.main()
