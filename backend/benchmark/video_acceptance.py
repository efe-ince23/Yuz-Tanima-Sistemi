import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
from uuid import UUID


@dataclass(frozen=True)
class TimeInterval:
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class ExpectedFace:
    face_id: UUID
    name: Optional[str]
    intervals: Tuple[TimeInterval, ...]


@dataclass(frozen=True)
class ExpectedAnonymousFace:
    label: str
    intervals: Tuple[TimeInterval, ...]


@dataclass(frozen=True)
class AcceptanceLimits:
    time_tolerance_ms: int = 1500
    minimum_identity_recall: float = 1.0
    minimum_temporal_iou: float = 0.35
    maximum_unexpected_known: int = 0
    minimum_anonymous_recall: float = 1.0
    minimum_anonymous_temporal_iou: float = 0.35
    minimum_anonymous_tracks: Optional[int] = None
    maximum_anonymous_tracks: Optional[int] = None
    maximum_tracks_per_expected_face: Optional[int] = None
    maximum_total_tracks: Optional[int] = None
    maximum_short_tracks: Optional[int] = None
    maximum_realtime_factor: Optional[float] = None


@dataclass(frozen=True)
class AcceptanceCase:
    case_id: str
    enabled: bool
    process_id: Optional[UUID]
    filename: Optional[str]
    expected_faces: Tuple[ExpectedFace, ...]
    expected_anonymous_faces: Tuple[ExpectedAnonymousFace, ...]
    evaluation_window: Optional[TimeInterval]
    limits: AcceptanceLimits


@dataclass(frozen=True)
class AcceptanceManifest:
    suite_name: str
    sample_fps: float
    cases: Tuple[AcceptanceCase, ...]


@dataclass(frozen=True)
class PredictedTrack:
    face_id: UUID
    status: str
    name: Optional[str]
    first_seen_ms: int
    last_seen_ms: int
    confidence: Optional[float]
    track_id: Optional[int] = None
    source_track_id: Optional[int] = None
    observation_count: Optional[int] = None
    detection_confidence: Optional[float] = None
    threshold: Optional[float] = None


def _track_diagnostics(
    predicted_tracks: Sequence[PredictedTrack],
    expected_ids: set,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, int]]:
    track_rows: List[Dict[str, object]] = []
    grouped: Dict[UUID, List[PredictedTrack]] = {}
    short_track_count = 0
    low_confidence_known_count = 0
    low_detection_count = 0

    for track in predicted_tracks:
        grouped.setdefault(track.face_id, []).append(track)
        duration_ms = max(0, track.last_seen_ms - track.first_seen_ms)
        flags: List[str] = []
        if duration_ms < 1000:
            flags.append("short_track")
            short_track_count += 1
        if (
            track.status == "known"
            and track.confidence is not None
            and track.threshold is not None
            and track.confidence < track.threshold + 0.08
        ):
            flags.append("low_known_margin")
            low_confidence_known_count += 1
        if (
            track.detection_confidence is not None
            and track.detection_confidence < 0.50
        ):
            flags.append("low_detection_confidence")
            low_detection_count += 1
        if track.status == "known" and track.face_id not in expected_ids:
            flags.append("unexpected_known_identity")

        track_rows.append(
            {
                "trackId": track.track_id,
                "sourceTrackId": track.source_track_id,
                "faceId": str(track.face_id),
                "status": track.status,
                "name": track.name,
                "startMs": track.first_seen_ms,
                "endMs": track.last_seen_ms,
                "durationMs": duration_ms,
                "observationCount": track.observation_count,
                "confidence": (
                    round(track.confidence, 6)
                    if track.confidence is not None
                    else None
                ),
                "detectionConfidence": (
                    round(track.detection_confidence, 6)
                    if track.detection_confidence is not None
                    else None
                ),
                "threshold": track.threshold,
                "flags": flags,
            }
        )

    identity_rows: List[Dict[str, object]] = []
    fragmented_identity_count = 0
    for face_id, tracks in sorted(grouped.items(), key=lambda item: str(item[0])):
        confidences = [
            track.confidence
            for track in tracks
            if track.confidence is not None
        ]
        is_fragmented = len(tracks) > 1
        if is_fragmented:
            fragmented_identity_count += 1
        identity_rows.append(
            {
                "faceId": str(face_id),
                "status": tracks[0].status,
                "name": next((track.name for track in tracks if track.name), None),
                "trackCount": len(tracks),
                "totalDurationMs": sum(
                    max(0, track.last_seen_ms - track.first_seen_ms)
                    for track in tracks
                ),
                "totalObservations": sum(
                    track.observation_count or 0 for track in tracks
                ),
                "averageConfidence": (
                    round(sum(confidences) / len(confidences), 6)
                    if confidences
                    else None
                ),
                "maximumConfidence": (
                    round(max(confidences), 6) if confidences else None
                ),
                "fragmented": is_fragmented,
                "intervals": [
                    {
                        "startMs": track.first_seen_ms,
                        "endMs": track.last_seen_ms,
                    }
                    for track in sorted(tracks, key=lambda item: item.first_seen_ms)
                ],
            }
        )

    risk_summary = {
        "shortTrackCount": short_track_count,
        "lowConfidenceKnownTrackCount": low_confidence_known_count,
        "lowDetectionTrackCount": low_detection_count,
        "fragmentedIdentityCount": fragmented_identity_count,
    }
    return track_rows, identity_rows, risk_summary


def _number(value: object, name: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} sayisal olmalidir.")
    result = float(value)
    if result < minimum:
        raise ValueError(f"{name} en az {minimum} olmalidir.")
    return result


def _ratio(value: object, name: str) -> float:
    result = _number(value, name)
    if result > 1:
        raise ValueError(f"{name} 0 ile 1 arasinda olmalidir.")
    return result


def _optional_number(
    values: Mapping[str, object],
    key: str,
    name: str,
) -> Optional[float]:
    value = values.get(key)
    if value is None:
        return None
    return _number(value, name)


def _optional_integer(
    values: Mapping[str, object],
    key: str,
    name: str,
) -> Optional[int]:
    value = values.get(key)
    if value is None:
        return None
    number = _number(value, name)
    if not number.is_integer():
        raise ValueError(f"{name} tam sayi olmalidir.")
    return int(number)


def _limits(values: Mapping[str, object]) -> AcceptanceLimits:
    tolerance = _number(
        values.get("timeToleranceSeconds", 1.5),
        "timeToleranceSeconds",
    )
    unexpected = _number(
        values.get("maximumUnexpectedKnown", 0),
        "maximumUnexpectedKnown",
    )
    if not unexpected.is_integer():
        raise ValueError("maximumUnexpectedKnown tam sayi olmalidir.")
    return AcceptanceLimits(
        time_tolerance_ms=round(tolerance * 1000),
        minimum_identity_recall=_ratio(
            values.get("minimumIdentityRecall", 1.0),
            "minimumIdentityRecall",
        ),
        minimum_temporal_iou=_ratio(
            values.get(
                "minimumTemporalIoU",
                values.get("minimumTemporalIou", 0.35),
            ),
            "minimumTemporalIoU",
        ),
        maximum_unexpected_known=int(unexpected),
        minimum_anonymous_recall=_ratio(
            values.get("minimumAnonymousRecall", 1.0),
            "minimumAnonymousRecall",
        ),
        minimum_anonymous_temporal_iou=_ratio(
            values.get("minimumAnonymousTemporalIoU", 0.35),
            "minimumAnonymousTemporalIoU",
        ),
        minimum_anonymous_tracks=_optional_integer(
            values,
            "minimumAnonymousTracks",
            "minimumAnonymousTracks",
        ),
        maximum_anonymous_tracks=_optional_integer(
            values,
            "maximumAnonymousTracks",
            "maximumAnonymousTracks",
        ),
        maximum_tracks_per_expected_face=_optional_integer(
            values,
            "maximumTracksPerExpectedFace",
            "maximumTracksPerExpectedFace",
        ),
        maximum_total_tracks=_optional_integer(
            values,
            "maximumTotalTracks",
            "maximumTotalTracks",
        ),
        maximum_short_tracks=_optional_integer(
            values,
            "maximumShortTracks",
            "maximumShortTracks",
        ),
        maximum_realtime_factor=_optional_number(
            values,
            "maximumRealtimeFactor",
            "maximumRealtimeFactor",
        ),
    )


def _merge_limits(
    defaults: Mapping[str, object],
    overrides: object,
) -> AcceptanceLimits:
    merged: Dict[str, object] = dict(defaults)
    if overrides is not None:
        if not isinstance(overrides, dict):
            raise ValueError("case limits bir nesne olmalidir.")
        merged.update(overrides)
    return _limits(merged)


def _interval(value: object, name: str) -> TimeInterval:
    if not isinstance(value, dict):
        raise ValueError(f"{name} bir nesne olmalidir.")
    start = _number(value.get("startSeconds"), f"{name}.startSeconds")
    end = _number(value.get("endSeconds"), f"{name}.endSeconds")
    if end < start:
        raise ValueError(f"{name} bitisi baslangictan once olamaz.")
    return TimeInterval(start_ms=round(start * 1000), end_ms=round(end * 1000))


def _expected_face(value: object, name: str) -> ExpectedFace:
    if not isinstance(value, dict):
        raise ValueError(f"{name} bir nesne olmalidir.")
    try:
        face_id = UUID(str(value.get("faceId")))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name}.faceId gecerli bir UUID olmalidir.") from error
    raw_name = value.get("name")
    if raw_name is not None and (not isinstance(raw_name, str) or not raw_name.strip()):
        raise ValueError(f"{name}.name bos olmayan metin olmalidir.")
    raw_intervals = value.get("intervals", [])
    if not isinstance(raw_intervals, list):
        raise ValueError(f"{name}.intervals bir liste olmalidir.")
    intervals = tuple(
        _interval(item, f"{name}.intervals[{index}]")
        for index, item in enumerate(raw_intervals)
    )
    return ExpectedFace(
        face_id=face_id,
        name=raw_name.strip() if isinstance(raw_name, str) else None,
        intervals=intervals,
    )


def _expected_anonymous_face(
    value: object,
    name: str,
) -> ExpectedAnonymousFace:
    if not isinstance(value, dict):
        raise ValueError(f"{name} bir nesne olmalidir.")
    label = value.get("label")
    if not isinstance(label, str) or not label.strip():
        raise ValueError(f"{name}.label bos olmayan metin olmalidir.")
    raw_intervals = value.get("intervals")
    if not isinstance(raw_intervals, list) or not raw_intervals:
        raise ValueError(f"{name}.intervals en az bir zaman araligi icermelidir.")
    return ExpectedAnonymousFace(
        label=label.strip(),
        intervals=tuple(
            _interval(item, f"{name}.intervals[{index}]")
            for index, item in enumerate(raw_intervals)
        ),
    )


def load_manifest(path: Path) -> AcceptanceManifest:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Manifest JSON okunamadi: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("Manifest kok degeri bir nesne olmalidir.")
    if payload.get("version") != 1:
        raise ValueError("Yalnizca video kabul manifest version=1 desteklenir.")
    suite_name = payload.get("suiteName", "Video Acceptance")
    if not isinstance(suite_name, str) or not suite_name.strip():
        raise ValueError("suiteName bos olmayan metin olmalidir.")
    sample_fps = _number(payload.get("sampleFps", 6), "sampleFps", 0.01)
    defaults = payload.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError("defaults bir nesne olmalidir.")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("Manifest en az bir case icermelidir.")

    cases: List[AcceptanceCase] = []
    case_ids = set()
    for index, value in enumerate(raw_cases):
        name = f"cases[{index}]"
        if not isinstance(value, dict):
            raise ValueError(f"{name} bir nesne olmalidir.")
        case_id = value.get("id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"{name}.id bos olmayan metin olmalidir.")
        case_id = case_id.strip()
        if case_id in case_ids:
            raise ValueError(f"Tekrarlanan case id: {case_id}")
        case_ids.add(case_id)
        enabled = value.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError(f"{name}.enabled boolean olmalidir.")
        source = value.get("source")
        if not isinstance(source, dict):
            raise ValueError(f"{name}.source bir nesne olmalidir.")
        raw_process_id = source.get("processId")
        filename = source.get("filename")
        process_id = None
        if raw_process_id is not None:
            try:
                process_id = UUID(str(raw_process_id))
            except (TypeError, ValueError) as error:
                raise ValueError(f"{name}.source.processId gecersiz.") from error
        if filename is not None and (
            not isinstance(filename, str) or not filename.strip()
        ):
            raise ValueError(f"{name}.source.filename gecersiz.")
        if process_id is None and filename is None:
            raise ValueError(f"{name}.source processId veya filename icermelidir.")
        raw_faces = value.get("expectedFaces", [])
        if not isinstance(raw_faces, list):
            raise ValueError(f"{name}.expectedFaces bir liste olmalidir.")
        faces = tuple(
            _expected_face(item, f"{name}.expectedFaces[{face_index}]")
            for face_index, item in enumerate(raw_faces)
        )
        if len({face.face_id for face in faces}) != len(faces):
            raise ValueError(f"{name} ayni faceId degerini birden cok kez iceriyor.")
        raw_anonymous_faces = value.get("expectedAnonymousFaces", [])
        if not isinstance(raw_anonymous_faces, list):
            raise ValueError(f"{name}.expectedAnonymousFaces bir liste olmalidir.")
        anonymous_faces = tuple(
            _expected_anonymous_face(
                item,
                f"{name}.expectedAnonymousFaces[{face_index}]",
            )
            for face_index, item in enumerate(raw_anonymous_faces)
        )
        raw_evaluation_window = value.get("evaluationWindow")
        evaluation_window = (
            _interval(raw_evaluation_window, f"{name}.evaluationWindow")
            if raw_evaluation_window is not None
            else None
        )
        cases.append(
            AcceptanceCase(
                case_id=case_id,
                enabled=enabled,
                process_id=process_id,
                filename=filename.strip() if isinstance(filename, str) else None,
                expected_faces=faces,
                expected_anonymous_faces=anonymous_faces,
                evaluation_window=evaluation_window,
                limits=_merge_limits(defaults, value.get("limits")),
            )
        )

    return AcceptanceManifest(
        suite_name=suite_name.strip(),
        sample_fps=sample_fps,
        cases=tuple(cases),
    )


def _interval_iou(
    expected: TimeInterval,
    predicted: TimeInterval,
    tolerance_ms: int,
) -> float:
    predicted_start = (
        expected.start_ms
        if abs(predicted.start_ms - expected.start_ms) <= tolerance_ms
        else predicted.start_ms
    )
    predicted_end = (
        expected.end_ms
        if abs(predicted.end_ms - expected.end_ms) <= tolerance_ms
        else predicted.end_ms
    )
    intersection = max(
        0,
        min(expected.end_ms, predicted_end)
        - max(expected.start_ms, predicted_start),
    )
    union = max(expected.end_ms, predicted_end) - min(
        expected.start_ms,
        predicted_start,
    )
    if union == 0:
        return 1.0 if expected.start_ms == predicted_start else 0.0
    return intersection / union


def _match_intervals(
    expected: Sequence[TimeInterval],
    predicted: Sequence[TimeInterval],
    tolerance_ms: int,
) -> Tuple[float, List[float]]:
    if not expected:
        return 1.0, []
    available = set(range(len(predicted)))
    scores: List[float] = []
    for expected_interval in expected:
        candidates = [
            (
                _interval_iou(
                    expected_interval,
                    predicted[predicted_index],
                    tolerance_ms,
                ),
                predicted_index,
            )
            for predicted_index in available
        ]
        if not candidates:
            scores.append(0.0)
            continue
        score, selected = max(candidates, key=lambda item: (item[0], -item[1]))
        available.remove(selected)
        scores.append(score)
    return sum(scores) / len(scores), scores


def _tracks_in_evaluation_window(
    tracks: Sequence[PredictedTrack],
    window: Optional[TimeInterval],
) -> Tuple[PredictedTrack, ...]:
    if window is None:
        return tuple(tracks)
    clipped = []
    for track in tracks:
        start_ms = max(track.first_seen_ms, window.start_ms)
        end_ms = min(track.last_seen_ms, window.end_ms)
        if end_ms < start_ms:
            continue
        clipped.append(
            replace(track, first_seen_ms=start_ms, last_seen_ms=end_ms)
        )
    return tuple(clipped)


def _anonymous_face_results(
    expected_faces: Sequence[ExpectedAnonymousFace],
    anonymous_tracks: Sequence[PredictedTrack],
    tolerance_ms: int,
    minimum_temporal_iou: float,
) -> Tuple[List[Dict[str, object]], float, float]:
    grouped: Dict[UUID, List[PredictedTrack]] = {}
    for track in anonymous_tracks:
        grouped.setdefault(track.face_id, []).append(track)

    available_ids = set(grouped)
    rows: List[Dict[str, object]] = []
    matched_count = 0
    scores: List[float] = []
    for expected in expected_faces:
        candidates = []
        for face_id in available_ids:
            predicted_intervals = [
                TimeInterval(track.first_seen_ms, track.last_seen_ms)
                for track in grouped[face_id]
            ]
            score, interval_scores = _match_intervals(
                expected.intervals,
                predicted_intervals,
                tolerance_ms,
            )
            candidates.append((score, face_id, interval_scores, predicted_intervals))

        if candidates:
            score, face_id, interval_scores, predicted_intervals = max(
                candidates,
                key=lambda item: (item[0], str(item[1])),
            )
            available_ids.remove(face_id)
        else:
            score = 0.0
            face_id = None
            interval_scores = [0.0 for _ in expected.intervals]
            predicted_intervals = []

        matched = score >= minimum_temporal_iou
        matched_count += int(matched)
        scores.append(score)
        rows.append(
            {
                "label": expected.label,
                "matched": matched,
                "matchedFaceId": str(face_id) if face_id is not None else None,
                "temporalIou": round(score, 6),
                "intervalScores": [round(item, 6) for item in interval_scores],
                "predictedIntervals": [
                    {"startMs": item.start_ms, "endMs": item.end_ms}
                    for item in predicted_intervals
                ],
            }
        )

    recall = matched_count / len(expected_faces) if expected_faces else 1.0
    temporal_iou = sum(scores) / len(scores) if scores else 1.0
    return rows, recall, temporal_iou


def evaluate_case(
    case: AcceptanceCase,
    predicted_tracks: Sequence[PredictedTrack],
    elapsed_seconds: float,
    duration_seconds: float,
) -> Dict[str, object]:
    predicted_tracks = _tracks_in_evaluation_window(
        predicted_tracks,
        case.evaluation_window,
    )
    expected_ids = {face.face_id for face in case.expected_faces}
    known_tracks = [track for track in predicted_tracks if track.status == "known"]
    anonymous_tracks = [track for track in predicted_tracks if track.status != "known"]
    predicted_known_ids = {track.face_id for track in known_tracks}
    matched_ids = expected_ids & predicted_known_ids
    identity_recall = len(matched_ids) / len(expected_ids) if expected_ids else 1.0
    unexpected_ids = sorted(predicted_known_ids - expected_ids, key=str)
    missed_ids = sorted(expected_ids - predicted_known_ids, key=str)
    track_rows, identity_rows, risk_summary = _track_diagnostics(
        predicted_tracks,
        expected_ids,
    )
    anonymous_face_results, anonymous_recall, anonymous_temporal_iou = (
        _anonymous_face_results(
            case.expected_anonymous_faces,
            anonymous_tracks,
            case.limits.time_tolerance_ms,
            case.limits.minimum_anonymous_temporal_iou,
        )
    )

    face_results = []
    temporal_scores: List[float] = []
    for expected_face in case.expected_faces:
        matching_tracks = [
            track for track in known_tracks if track.face_id == expected_face.face_id
        ]
        predicted_intervals = [
            TimeInterval(track.first_seen_ms, track.last_seen_ms)
            for track in matching_tracks
        ]
        temporal_iou, interval_scores = _match_intervals(
            expected_face.intervals,
            predicted_intervals,
            case.limits.time_tolerance_ms,
        )
        if expected_face.intervals:
            temporal_scores.append(temporal_iou)
        face_results.append(
            {
                "faceId": str(expected_face.face_id),
                "name": expected_face.name,
                "recognized": bool(matching_tracks),
                "trackCount": len(matching_tracks),
                "temporalIou": round(temporal_iou, 6),
                "intervalScores": [round(score, 6) for score in interval_scores],
                "predictedIntervals": [
                    {
                        "startMs": interval.start_ms,
                        "endMs": interval.end_ms,
                    }
                    for interval in predicted_intervals
                ],
            }
        )
    temporal_iou = (
        sum(temporal_scores) / len(temporal_scores) if temporal_scores else 1.0
    )
    realtime_factor = (
        elapsed_seconds / duration_seconds if duration_seconds > 0 else None
    )
    checks = [
        {
            "name": "identity_recall",
            "passed": identity_recall >= case.limits.minimum_identity_recall,
            "actual": round(identity_recall, 6),
            "expected": f">={case.limits.minimum_identity_recall}",
        },
        {
            "name": "temporal_iou",
            "passed": temporal_iou >= case.limits.minimum_temporal_iou,
            "actual": round(temporal_iou, 6),
            "expected": f">={case.limits.minimum_temporal_iou}",
        },
        {
            "name": "unexpected_known",
            "passed": len(unexpected_ids) <= case.limits.maximum_unexpected_known,
            "actual": len(unexpected_ids),
            "expected": f"<={case.limits.maximum_unexpected_known}",
        },
    ]
    if case.expected_anonymous_faces:
        checks.extend(
            [
                {
                    "name": "anonymous_recall",
                    "passed": anonymous_recall
                    >= case.limits.minimum_anonymous_recall,
                    "actual": round(anonymous_recall, 6),
                    "expected": f">={case.limits.minimum_anonymous_recall}",
                },
                {
                    "name": "anonymous_temporal_iou",
                    "passed": anonymous_temporal_iou
                    >= case.limits.minimum_anonymous_temporal_iou,
                    "actual": round(anonymous_temporal_iou, 6),
                    "expected": (
                        f">={case.limits.minimum_anonymous_temporal_iou}"
                    ),
                },
            ]
        )
    if case.limits.minimum_anonymous_tracks is not None:
        checks.append(
            {
                "name": "minimum_anonymous_tracks",
                "passed": len(anonymous_tracks)
                >= case.limits.minimum_anonymous_tracks,
                "actual": len(anonymous_tracks),
                "expected": f">={case.limits.minimum_anonymous_tracks}",
            }
        )
    if case.limits.maximum_anonymous_tracks is not None:
        checks.append(
            {
                "name": "anonymous_tracks",
                "passed": len(anonymous_tracks)
                <= case.limits.maximum_anonymous_tracks,
                "actual": len(anonymous_tracks),
                "expected": f"<={case.limits.maximum_anonymous_tracks}",
            }
        )
    if case.limits.maximum_total_tracks is not None:
        checks.append(
            {
                "name": "total_tracks",
                "passed": len(predicted_tracks)
                <= case.limits.maximum_total_tracks,
                "actual": len(predicted_tracks),
                "expected": f"<={case.limits.maximum_total_tracks}",
            }
        )
    if case.limits.maximum_short_tracks is not None:
        checks.append(
            {
                "name": "short_tracks",
                "passed": risk_summary["shortTrackCount"]
                <= case.limits.maximum_short_tracks,
                "actual": risk_summary["shortTrackCount"],
                "expected": f"<={case.limits.maximum_short_tracks}",
            }
        )
    if case.limits.maximum_tracks_per_expected_face is not None:
        maximum_tracks = max(
            (int(item["trackCount"]) for item in face_results),
            default=0,
        )
        checks.append(
            {
                "name": "tracks_per_expected_face",
                "passed": maximum_tracks
                <= case.limits.maximum_tracks_per_expected_face,
                "actual": maximum_tracks,
                "expected": f"<={case.limits.maximum_tracks_per_expected_face}",
            }
        )
    if (
        case.limits.maximum_realtime_factor is not None
        and realtime_factor is not None
    ):
        checks.append(
            {
                "name": "realtime_factor",
                "passed": realtime_factor
                <= case.limits.maximum_realtime_factor,
                "actual": round(realtime_factor, 6),
                "expected": f"<={case.limits.maximum_realtime_factor}",
            }
        )

    return {
        "id": case.case_id,
        "status": "passed" if all(bool(item["passed"]) for item in checks) else "failed",
        "metrics": {
            "expectedKnownFaces": len(expected_ids),
            "matchedKnownFaces": len(matched_ids),
            "identityRecall": round(identity_recall, 6),
            "temporalIou": round(temporal_iou, 6),
            "expectedAnonymousFaces": len(case.expected_anonymous_faces),
            "anonymousRecall": round(anonymous_recall, 6),
            "anonymousTemporalIou": round(anonymous_temporal_iou, 6),
            "unexpectedKnownCount": len(unexpected_ids),
            "anonymousTrackCount": len(anonymous_tracks),
            "totalTrackCount": len(predicted_tracks),
            "processingSeconds": round(elapsed_seconds, 3),
            "videoDurationSeconds": round(duration_seconds, 3),
            "realtimeFactor": (
                round(realtime_factor, 6) if realtime_factor is not None else None
            ),
            **risk_summary,
        },
        "missedFaceIds": [str(face_id) for face_id in missed_ids],
        "unexpectedKnownFaceIds": [str(face_id) for face_id in unexpected_ids],
        "faces": face_results,
        "anonymousFaces": anonymous_face_results,
        "evaluationWindow": (
            {
                "startMs": case.evaluation_window.start_ms,
                "endMs": case.evaluation_window.end_ms,
            }
            if case.evaluation_window is not None
            else None
        ),
        "trackDiagnostics": track_rows,
        "identityDiagnostics": identity_rows,
        "checks": checks,
    }
