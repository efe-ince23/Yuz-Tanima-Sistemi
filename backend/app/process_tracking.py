import json
import logging
import os
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import func

from app.database import SessionLocal
from app.face_storage import DATA_ROOT
from app.models import RecognitionProcess


logger = logging.getLogger(__name__)
TRACKED_FACE_PATHS = {
    "/faces/recognize": "identify",
    "/api/faces/detect": "detect",
    "/api/faces/compare": "compare",
    "/api/faces/identify": "identify",
    "/api/videos": "video_recognize",
    "/api/videos/live-recordings": "video_recognize",
}
PROCESS_LOG_ROOT = DATA_ROOT / "logs"
PROCESS_FALLBACK_LOG = PROCESS_LOG_ROOT / "recognition-processes.jsonl"
_fallback_log_lock = Lock()


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _task_detail(
    operation_type: str,
    status: str,
    face_count: int,
    faces: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    return {
        "operation_type": operation_type,
        "processed_face_count": face_count,
        "faces": faces or [],
        "status": status,
    }


def _write_fallback_log(payload: Dict[str, Any]) -> bool:
    try:
        PROCESS_LOG_ROOT.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        with _fallback_log_lock:
            with PROCESS_FALLBACK_LOG.open("a", encoding="utf-8") as log_file:
                log_file.write(serialized + "\n")
                log_file.flush()
                os.fsync(log_file.fileno())
        return True
    except Exception:
        logger.exception("Fallback process log could not be written")
        return False


def read_fallback_process(process_id: UUID) -> Optional[Dict[str, Any]]:
    if not PROCESS_FALLBACK_LOG.exists():
        return None

    latest = None
    try:
        with _fallback_log_lock:
            with PROCESS_FALLBACK_LOG.open("r", encoding="utf-8") as log_file:
                for line in log_file:
                    try:
                        payload = json.loads(line)
                    except (TypeError, ValueError):
                        continue
                    if payload.get("process_id") == str(process_id):
                        latest = payload
    except OSError:
        logger.exception("Fallback process log could not be read")
    return latest


def begin_process(
    process_id: UUID,
    operation_type: str,
    owner_user_id: Optional[UUID] = None,
) -> bool:
    created_at = _timestamp()
    try:
        with SessionLocal() as session:
            session.add(
                RecognitionProcess(
                    process_id=process_id,
                    operation_type=operation_type,
                    owner_user_id=owner_user_id,
                    status="processing",
                    face_count=0,
                    task_detail=_task_detail(operation_type, "processing", 0),
                )
            )
            session.commit()
        return True
    except Exception:
        logger.exception("Recognition process could not be created: %s", process_id)
        _write_fallback_log(
            {
                "process_id": str(process_id),
                "operation_type": operation_type,
                "status": "processing",
                "http_status": None,
                "face_count": 0,
                "task_detail": _task_detail(operation_type, "processing", 0),
                "result": None,
                "error_detail": None,
                "created_at": created_at,
                "completed_at": None,
            }
        )
        return False


def complete_process(
    process_id: UUID,
    *,
    operation_type: str,
    status: str,
    http_status: int,
    face_count: int,
    faces: Optional[List[Dict[str, Any]]],
    result: Optional[Dict[str, Any]],
) -> bool:
    task_detail = _task_detail(operation_type, status, face_count, faces)
    completed_at = _timestamp()
    try:
        with SessionLocal() as session:
            process = session.get(RecognitionProcess, process_id)
            if process is None:
                raise RuntimeError("Recognition process was not created.")
            process.status = status
            process.http_status = http_status
            process.face_count = face_count
            process.task_detail = task_detail
            process.result = result
            process.error_detail = None
            process.completed_at = func.now()
            session.commit()
        return True
    except Exception:
        logger.exception("Recognition process could not be completed: %s", process_id)
        previous = read_fallback_process(process_id)
        _write_fallback_log(
            {
                "process_id": str(process_id),
                "operation_type": operation_type,
                "status": status,
                "http_status": http_status,
                "face_count": face_count,
                "task_detail": task_detail,
                "result": result,
                "error_detail": None,
                "created_at": (previous or {}).get("created_at", completed_at),
                "completed_at": completed_at,
            }
        )
        return False


def fail_process(
    process_id: UUID,
    operation_type: str,
    http_status: int,
    error_detail: str,
) -> bool:
    task_detail = _task_detail(operation_type, "failed", 0)
    completed_at = _timestamp()
    try:
        with SessionLocal() as session:
            process = session.get(RecognitionProcess, process_id)
            if process is None:
                raise RuntimeError("Recognition process was not created.")
            if process.status != "processing":
                return True
            process.status = "failed"
            process.http_status = http_status
            process.face_count = 0
            process.task_detail = task_detail
            process.error_detail = error_detail[:4000]
            process.completed_at = func.now()
            session.commit()
        return True
    except Exception:
        logger.exception("Recognition process failure could not be saved: %s", process_id)
        previous = read_fallback_process(process_id)
        _write_fallback_log(
            {
                "process_id": str(process_id),
                "operation_type": operation_type,
                "status": "failed",
                "http_status": http_status,
                "face_count": 0,
                "task_detail": task_detail,
                "result": None,
                "error_detail": error_detail[:4000],
                "created_at": (previous or {}).get("created_at", completed_at),
                "completed_at": completed_at,
            }
        )
        return False


def complete_process_if_pending(
    process_id: UUID,
    operation_type: str,
    http_status: int,
) -> None:
    fallback = read_fallback_process(process_id)
    if fallback is not None and fallback.get("status") != "processing":
        return
    try:
        with SessionLocal() as session:
            process = session.get(RecognitionProcess, process_id)
            if process is not None and process.status != "processing":
                return
    except Exception:
        logger.exception("Pending process state could not be checked: %s", process_id)
    complete_process(
        process_id,
        operation_type=operation_type,
        status="completed",
        http_status=http_status,
        face_count=0,
        faces=[],
        result=None,
    )
