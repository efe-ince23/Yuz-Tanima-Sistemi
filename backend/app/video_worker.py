import logging
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Optional, Set
from uuid import UUID

from sqlalchemy import select

from app.database import SessionLocal
from app.models import VideoJob
from app.video_config import get_video_settings
from app.video_processing import VideoProcessingError, process_video_job


logger = logging.getLogger(__name__)


class VideoWorkerPool:
    def __init__(self) -> None:
        self._lock = Lock()
        self._executor: Optional[ThreadPoolExecutor] = None
        self._submitted: Set[UUID] = set()

    def start(self) -> None:
        with self._lock:
            if self._executor is not None:
                return
            concurrency = get_video_settings().processing_concurrency
            self._executor = ThreadPoolExecutor(
                max_workers=concurrency,
                thread_name_prefix="video-worker",
            )

        recovered = self._recover_pending_jobs()
        for process_id in recovered:
            self.submit(process_id)
        logger.info(
            "Video worker started: concurrency=%s recovered=%s",
            concurrency,
            len(recovered),
        )

    def stop(self) -> None:
        with self._lock:
            executor = self._executor
            self._executor = None
        if executor is not None:
            executor.shutdown(wait=True)
        logger.info("Video worker stopped")

    def submit(self, process_id: UUID) -> bool:
        with self._lock:
            if self._executor is None:
                raise RuntimeError("Video worker henuz baslatilmadi.")
            if process_id in self._submitted:
                return False
            self._submitted.add(process_id)
            try:
                self._executor.submit(self._run, process_id)
            except Exception:
                self._submitted.discard(process_id)
                raise
        return True

    def _recover_pending_jobs(self) -> list:
        with SessionLocal() as session:
            jobs = list(
                session.scalars(
                    select(VideoJob)
                    .where(VideoJob.status.in_(("queued", "processing")))
                    .order_by(VideoJob.created_at)
                ).all()
            )
            for job in jobs:
                if job.status == "processing":
                    job.status = "queued"
                    job.progress_percent = 0.0
                    job.error_code = None
                    job.error_detail = None
                    job.process.status = "queued"
                    job.process.http_status = 202
                    job.process.error_detail = None
                    if job.process.task_detail is not None:
                        job.process.task_detail = {
                            **job.process.task_detail,
                            "status": "queued",
                        }
            session.commit()
            return [job.process_id for job in jobs]

    def _run(self, process_id: UUID) -> None:
        try:
            with SessionLocal() as session:
                process_video_job(session, process_id)
            logger.info("Video processing completed: %s", process_id)
        except VideoProcessingError as error:
            logger.error(
                "Video processing failed: process_id=%s code=%s detail=%s",
                process_id,
                error.code,
                error.message,
            )
        except Exception:
            logger.exception("Unexpected video worker failure: %s", process_id)
        finally:
            with self._lock:
                self._submitted.discard(process_id)


video_worker_pool = VideoWorkerPool()


def start_video_workers() -> None:
    video_worker_pool.start()


def stop_video_workers() -> None:
    video_worker_pool.stop()


def submit_video_job(process_id: UUID) -> bool:
    return video_worker_pool.submit(process_id)
