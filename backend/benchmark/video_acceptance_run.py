import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

import numpy as np
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models import VideoJob
from app.video_recognition import recognize_video_tracks
from app.yolo_arcface import get_yolo_arcface_engine
from benchmark.video_acceptance import (
    AcceptanceCase,
    PredictedTrack,
    evaluate_case,
    load_manifest,
)
from benchmark.video_acceptance_report import write_video_acceptance_reports


DEFAULT_MANIFEST = Path("/artifacts/video-acceptance/manifest.json")
DEFAULT_OUTPUT_ROOT = Path(
    os.getenv("VIDEO_ACCEPTANCE_OUTPUT_ROOT", "/artifacts/video-acceptance/runs")
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only acceptance tests for the video face recognition pipeline."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--create-draft",
        type=Path,
        help="Create a disabled manifest draft from recent completed video jobs.",
    )
    parser.add_argument("--draft-limit", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    return parser


def _begin_read_only(session: Session) -> None:
    session.execute(text("SET TRANSACTION READ ONLY"))


def _resolve_job(session: Session, case: AcceptanceCase) -> VideoJob:
    if case.process_id is not None:
        job = session.get(VideoJob, case.process_id)
    else:
        job = session.scalar(
            select(VideoJob)
            .where(
                VideoJob.original_filename == case.filename,
                VideoJob.status == "completed",
            )
            .order_by(VideoJob.created_at.desc())
            .limit(1)
        )
    if job is None:
        raise LookupError("Manifestte belirtilen video kaydi bulunamadi.")
    if job.status != "completed":
        raise ValueError("Kabul testinde yalnizca tamamlanmis video kullanilabilir.")
    return job


def _warm_video_models() -> float:
    started = time.perf_counter()
    engine = get_yolo_arcface_engine()

    # Session creation and first CUDA/TensorRT execution are intentionally kept
    # outside the video processing timer so benchmark runs remain comparable.
    engine.detect(np.full((640, 640, 3), 114, dtype=np.uint8))
    engine.recognizer.embed_aligned(
        [np.full((112, 112, 3), 127, dtype=np.uint8)]
    )
    return time.perf_counter() - started


def _run_case(case: AcceptanceCase, sample_fps: float) -> Dict[str, object]:
    warmup_seconds = _warm_video_models()
    session = SessionLocal()
    started = time.perf_counter()
    try:
        _begin_read_only(session)
        job = _resolve_job(session, case)
        summary = recognize_video_tracks(
            session,
            job.object_path,
            sample_fps,
            manage_transaction=False,
            read_only=True,
            owner_user_id=job.process.owner_user_id,
        )
        elapsed = time.perf_counter() - started
        duration_seconds = (
            float(job.duration_seconds)
            if job.duration_seconds is not None
            else summary.tracking.detection.last_timestamp_ms / 1000.0
        )
        predicted_tracks = tuple(
            PredictedTrack(
                face_id=track.face_id,
                status=track.status,
                name=track.name,
                first_seen_ms=track.first_seen_ms
                if track.first_seen_ms is not None
                else track.representative_timestamp_ms,
                last_seen_ms=track.last_seen_ms
                if track.last_seen_ms is not None
                else track.representative_timestamp_ms,
                confidence=track.similarity,
                track_id=track.track_id,
                source_track_id=track.source_track_id,
                observation_count=track.observation_count,
                detection_confidence=track.detection_confidence,
                threshold=track.threshold,
            )
            for track in summary.tracks
        )
        result = evaluate_case(
            case,
            predicted_tracks,
            elapsed_seconds=elapsed,
            duration_seconds=duration_seconds,
        )
        result["source"] = {
            "processId": str(job.process_id),
            "filename": job.original_filename,
        }
        result["pipeline"] = {
            "sampleFps": sample_fps,
            "sampledFrameCount": summary.tracking.detection.sampled_frame_count,
            "detectedFaceCount": summary.tracking.detection.detected_face_count,
            "spatialTrackCount": summary.tracking.unique_track_count,
            "ownerUserId": (
                str(job.process.owner_user_id)
                if job.process.owner_user_id is not None
                else None
            ),
            "modelWarmupSeconds": round(warmup_seconds, 3),
            "analysisSeconds": round(elapsed, 3),
            "endToEndSeconds": round(warmup_seconds + elapsed, 3),
            "samplingAndTrackingSeconds": round(summary.tracking_seconds, 3),
            "modelInferenceSeconds": round(
                summary.tracking.detection.inference_seconds,
                3,
            ),
            "faceDetectionSeconds": round(
                summary.tracking.detection.detector_seconds,
                3,
            ),
            "faceRecognitionSeconds": round(
                summary.tracking.detection.recognizer_seconds,
                3,
            ),
            "downloadAndDecodeSeconds": round(
                max(
                    0.0,
                    summary.tracking.detection.sampling_seconds
                    - summary.tracking.detection.inference_seconds,
                ),
                3,
            ),
            "identitySeconds": round(summary.identity_seconds, 3),
        }
        result["metrics"]["modelWarmupSeconds"] = round(warmup_seconds, 3)
        result["metrics"]["endToEndSeconds"] = round(
            warmup_seconds + elapsed,
            3,
        )
        return result
    finally:
        session.rollback()
        session.close()


def _error_case(case: AcceptanceCase, error: Exception) -> Dict[str, object]:
    return {
        "id": case.case_id,
        "status": "error",
        "source": {
            "processId": str(case.process_id) if case.process_id else None,
            "filename": case.filename,
        },
        "error": str(error),
        "checks": [],
    }


def run_manifest(manifest_path: Path, output_root: Path) -> Dict[str, object]:
    manifest = load_manifest(manifest_path)
    case_results = []
    for case in manifest.cases:
        if not case.enabled:
            continue
        print(f"  Video kabul testi: {case.case_id}", flush=True)
        try:
            case_results.append(_run_case(case, manifest.sample_fps))
        except Exception as error:
            case_results.append(_error_case(case, error))
            print(f"    Hata: {error}", file=sys.stderr, flush=True)

    passed = sum(item["status"] == "passed" for item in case_results)
    failed = len(case_results) - passed
    status = (
        "not_run"
        if not case_results
        else "passed"
        if failed == 0
        else "failed"
    )
    now = datetime.now(timezone.utc)
    result: Dict[str, object] = {
        "suiteName": manifest.suite_name,
        "createdAt": now.isoformat(),
        "status": status,
        "readOnly": True,
        "sampleFps": manifest.sample_fps,
        "summary": {
            "configuredCases": len(manifest.cases),
            "executedCases": len(case_results),
            "passedCases": passed,
            "failedCases": failed,
        },
        "cases": case_results,
    }
    run_directory = output_root / (
        now.strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    )
    report_paths = write_video_acceptance_reports(run_directory, result)
    result["reports"] = dict(report_paths)
    print(json.dumps({"status": status, "reports": report_paths}, indent=2))
    return result


def create_draft(path: Path, limit: int, force: bool = False) -> Path:
    if limit <= 0:
        raise ValueError("draft-limit sifirdan buyuk olmalidir.")
    if path.exists() and not force:
        raise FileExistsError(f"Manifest zaten var: {path}")
    session = SessionLocal()
    try:
        _begin_read_only(session)
        recent_jobs = session.scalars(
            select(VideoJob)
            .where(VideoJob.status == "completed")
            .order_by(VideoJob.created_at.desc())
        ).all()
        jobs = []
        filenames = set()
        for job in recent_jobs:
            if job.original_filename in filenames:
                continue
            filenames.add(job.original_filename)
            jobs.append(job)
            if len(jobs) >= limit:
                break
        if not jobs:
            raise LookupError("Taslak icin tamamlanmis video bulunamadi.")
        payload = {
            "version": 1,
            "suiteName": "Video Kabul Testleri",
            "sampleFps": 6,
            "defaults": {
                "timeToleranceSeconds": 1.5,
                "minimumIdentityRecall": 1.0,
                "minimumTemporalIoU": 0.35,
                "maximumUnexpectedKnown": 0,
                "minimumAnonymousRecall": 1.0,
                "minimumAnonymousTemporalIoU": 0.35,
                "minimumAnonymousTracks": None,
                "maximumAnonymousTracks": None,
                "maximumTracksPerExpectedFace": None,
                "maximumTotalTracks": None,
                "maximumShortTracks": None,
                "maximumRealtimeFactor": None,
            },
            "cases": [
                {
                    "id": f"video-{index}",
                    "enabled": False,
                    "source": {
                        "processId": str(job.process_id),
                        "filename": job.original_filename,
                    },
                    "expectedFaces": [],
                }
                for index, job in enumerate(jobs, start=1)
            ],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path
    finally:
        session.rollback()
        session.close()


def main(argv: Optional[List[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.create_draft is not None:
            path = create_draft(
                arguments.create_draft,
                arguments.draft_limit,
                arguments.force,
            )
            print(f"Taslak olusturuldu: {path}")
            return 0
        result = run_manifest(arguments.manifest, arguments.output_root)
        return 0 if result["status"] in ("passed", "not_run") else 2
    except Exception as error:
        print(f"Video kabul testi baslatilamadi: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
