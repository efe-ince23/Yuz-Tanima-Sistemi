import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models import VideoJob
from app.video_recognition import RecognizedVideoTrack, recognize_video_tracks


DEFAULT_OUTPUT = Path("/artifacts/video-acceptance/regression-baseline.json")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a read-only regression snapshot from completed videos."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--process-id", action="append", required=True)
    parser.add_argument("--sample-fps", type=float, default=6.0)
    parser.add_argument("--force", action="store_true")
    return parser


def _bounds(track: RecognizedVideoTrack):
    start_ms = (
        track.first_seen_ms
        if track.first_seen_ms is not None
        else track.representative_timestamp_ms
    )
    end_ms = (
        track.last_seen_ms
        if track.last_seen_ms is not None
        else track.representative_timestamp_ms
    )
    return start_ms, end_ms


def _interval(track: RecognizedVideoTrack) -> Dict[str, float]:
    start_ms, end_ms = _bounds(track)
    return {
        "startSeconds": round(start_ms / 1000.0, 3),
        "endSeconds": round(end_ms / 1000.0, 3),
    }


def _case_payload(
    index: int,
    job: VideoJob,
    tracks: Sequence[RecognizedVideoTrack],
) -> Dict[str, object]:
    known_groups: Dict[UUID, List[RecognizedVideoTrack]] = defaultdict(list)
    anonymous_groups: Dict[UUID, List[RecognizedVideoTrack]] = defaultdict(list)
    for track in tracks:
        target = known_groups if track.status == "known" else anonymous_groups
        target[track.face_id].append(track)

    expected_faces = []
    for face_id, grouped_tracks in sorted(
        known_groups.items(),
        key=lambda item: str(item[0]),
    ):
        expected_faces.append(
            {
                "faceId": str(face_id),
                "name": next(
                    (track.name for track in grouped_tracks if track.name),
                    None,
                ),
                "intervals": [
                    _interval(track)
                    for track in sorted(
                        grouped_tracks,
                        key=lambda item: item.first_seen_ms or 0,
                    )
                ],
            }
        )

    expected_anonymous = []
    for anonymous_index, (_, grouped_tracks) in enumerate(
        sorted(anonymous_groups.items(), key=lambda item: str(item[0])),
        start=1,
    ):
        expected_anonymous.append(
            {
                "label": f"Anonim yuz {anonymous_index}",
                "intervals": [
                    _interval(track)
                    for track in sorted(
                        grouped_tracks,
                        key=lambda item: item.first_seen_ms or 0,
                    )
                ],
            }
        )

    short_tracks = sum(
        1
        for track in tracks
        if _bounds(track)[1] - _bounds(track)[0] < 1000
    )
    known_track_counts = [len(group) for group in known_groups.values()]
    anonymous_track_count = sum(len(group) for group in anonymous_groups.values())
    return {
        "id": f"regression-{index}",
        "enabled": True,
        "source": {
            "processId": str(job.process_id),
            "filename": job.original_filename,
        },
        "expectedFaces": expected_faces,
        "expectedAnonymousFaces": expected_anonymous,
        "limits": {
            "minimumAnonymousTracks": anonymous_track_count,
            "maximumAnonymousTracks": anonymous_track_count,
            "maximumTracksPerExpectedFace": (
                max(known_track_counts) if known_track_counts else None
            ),
            "maximumTotalTracks": len(tracks),
            "maximumShortTracks": short_tracks,
        },
    }


def create_snapshot(
    output: Path,
    process_ids: Sequence[UUID],
    sample_fps: float,
    force: bool = False,
) -> Path:
    if sample_fps <= 0:
        raise ValueError("sample-fps sifirdan buyuk olmalidir.")
    if output.exists() and not force:
        raise FileExistsError(f"Snapshot zaten var: {output}")

    cases = []
    session: Session = SessionLocal()
    try:
        session.execute(text("SET TRANSACTION READ ONLY"))
        for index, process_id in enumerate(process_ids, start=1):
            job = session.get(VideoJob, process_id)
            if job is None or job.status != "completed":
                raise LookupError(f"Tamamlanmis video bulunamadi: {process_id}")
            summary = recognize_video_tracks(
                session,
                job.object_path,
                sample_fps,
                manage_transaction=False,
                read_only=True,
                owner_user_id=job.process.owner_user_id,
            )
            cases.append(_case_payload(index, job, summary.tracks))
    finally:
        session.rollback()
        session.close()

    payload = {
        "version": 1,
        "baselineType": "generated-regression-snapshot",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "suiteName": "Video Coklu Regresyon Seti",
        "sampleFps": sample_fps,
        "defaults": {
            "timeToleranceSeconds": 0.5,
            "minimumIdentityRecall": 1.0,
            "minimumTemporalIoU": 0.8,
            "maximumUnexpectedKnown": 0,
            "minimumAnonymousRecall": 1.0,
            "minimumAnonymousTemporalIoU": 0.8,
            "maximumRealtimeFactor": 1.0,
        },
        "cases": cases,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output


def main(argv: Optional[List[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        process_ids = [UUID(value) for value in arguments.process_id]
        path = create_snapshot(
            arguments.output,
            process_ids,
            arguments.sample_fps,
            arguments.force,
        )
        print(f"Regresyon snapshot olusturuldu: {path}")
        return 0
    except Exception as error:
        print(f"Regresyon snapshot olusturulamadi: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
