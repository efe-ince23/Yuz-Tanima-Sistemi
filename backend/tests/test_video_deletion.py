import unittest
from unittest.mock import MagicMock, patch
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy.exc import SQLAlchemyError

from app.models import RecognitionProcess, VideoJob
from app.routers.videos import delete_video_job


class VideoDeletionTests(unittest.TestCase):
    def setUp(self):
        self.process_id = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")

    def _job(self, status="completed"):
        return VideoJob(
            process_id=self.process_id,
            status=status,
            original_filename="test.mp4",
            object_path=f"videos/{self.process_id}/source.mp4",
            content_type="video/mp4",
            file_size_bytes=100,
        )

    def test_active_video_cannot_be_deleted(self):
        session = MagicMock()
        session.get.return_value = self._job("processing")

        with self.assertRaises(HTTPException) as raised:
            delete_video_job(self.process_id, session)

        self.assertEqual(raised.exception.status_code, 409)
        session.delete.assert_not_called()

    @patch("app.routers.videos.finalize_staged_files")
    @patch("app.routers.videos.stage_face_images_for_deletion")
    def test_completed_video_and_process_are_deleted(self, stage, finalize):
        job = self._job()
        process = RecognitionProcess(
            process_id=self.process_id,
            operation_type="video_recognize",
            status="completed",
        )
        session = MagicMock()
        session.get.side_effect = lambda model, _id: (
            job if model is VideoJob else process
        )
        staged = [MagicMock()]
        stage.return_value = staged

        response = delete_video_job(self.process_id, session)

        self.assertEqual(response.status_code, 204)
        stage.assert_called_once_with([job.object_path])
        session.delete.assert_called_once_with(process)
        session.commit.assert_called_once_with()
        finalize.assert_called_once_with(staged)

    @patch("app.routers.videos.restore_staged_files")
    @patch("app.routers.videos.stage_face_images_for_deletion")
    def test_database_failure_restores_staged_video(self, stage, restore):
        job = self._job()
        process = RecognitionProcess(
            process_id=self.process_id,
            operation_type="video_recognize",
            status="completed",
        )
        session = MagicMock()
        session.get.side_effect = lambda model, _id: (
            job if model is VideoJob else process
        )
        session.commit.side_effect = SQLAlchemyError("test")
        staged = [MagicMock()]
        stage.return_value = staged

        with self.assertRaises(HTTPException) as raised:
            delete_video_job(self.process_id, session)

        self.assertEqual(raised.exception.status_code, 500)
        session.rollback.assert_called_once_with()
        restore.assert_called_once_with(staged)


if __name__ == "__main__":
    unittest.main()
