import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
from uuid import uuid4

import cv2
import numpy as np

from app.yolo_arcface import (
    ARCFACE_BATCH_SIZE,
    ARCFACE_MODEL_PATH,
    YOLO_MODEL_PATH,
    get_yolo_arcface_engine,
)
from benchmark.dataset import (
    IdentitySplit,
    VerificationPair,
    collect_identity_splits,
    load_verification_pairs,
    unique_image_paths,
)
from benchmark.metrics import (
    best_threshold,
    metric_to_dict,
    normalized_rows,
    percentile,
    roc_auc,
    threshold_curve,
)
from benchmark.report import write_reports


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only LFW benchmark for YOLOv8-Face and ArcFace R50."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            os.getenv(
                "LFW_DATASET_ROOT",
                "/datasets/lfw/lfw-deepfunneled/lfw-deepfunneled",
            )
        ),
    )
    parser.add_argument(
        "--pairs-file",
        type=Path,
        default=Path(os.getenv("LFW_PAIRS_FILE", "/datasets/lfw/pairs.csv")),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(os.getenv("BENCHMARK_OUTPUT_ROOT", "/artifacts/benchmarks")),
    )
    parser.add_argument("--max-pairs-per-class", type=int, default=500)
    parser.add_argument("--max-identities", type=int, default=200)
    parser.add_argument("--max-unknowns", type=int, default=200)
    parser.add_argument("--performance-samples", type=int, default=96)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use all 3000 genuine/impostor LFW pairs and a larger identity split.",
    )
    return parser


def _positive(name: str, value: int) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")
    return value


def _validate_paths(dataset_root: Path, pairs_file: Path) -> None:
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"LFW dataset directory was not found: {dataset_root}")
    if not pairs_file.is_file():
        raise FileNotFoundError(f"LFW pairs file was not found: {pairs_file}")


def _sha256(path: str) -> Optional[str]:
    model_path = Path(path)
    if not model_path.is_file():
        return None
    digest = hashlib.sha256()
    with model_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gpu_info() -> Mapping[str, object]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return {"available": False}
    first_line = completed.stdout.strip().splitlines()[0]
    values = [value.strip() for value in first_line.split(",")]
    return {
        "available": True,
        "name": values[0] if values else None,
        "driver_version": values[1] if len(values) > 1 else None,
        "memory_total_mb": int(values[2]) if len(values) > 2 else None,
    }


def _process_images(paths: Sequence[Path], engine):
    aligned_faces: Dict[str, np.ndarray] = {}
    detection_times: List[float] = []
    face_counts = {"zero": 0, "single": 0, "multiple": 0, "decode_error": 0}
    total = len(paths)
    for index, path in enumerate(paths, start=1):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            face_counts["decode_error"] += 1
            continue
        started = time.perf_counter()
        faces = engine.detector.detect(image)
        detection_times.append((time.perf_counter() - started) * 1000.0)
        if not faces:
            face_counts["zero"] += 1
        elif len(faces) == 1:
            face_counts["single"] += 1
        else:
            face_counts["multiple"] += 1
        if faces:
            target_face = _select_lfw_target_face(image, faces)
            aligned_faces[str(path)] = engine.recognizer.align(image, target_face)
        if index == total or index % 100 == 0:
            print(f"  Detection: {index}/{total}", flush=True)

    embeddings: Dict[str, np.ndarray] = {}
    aligned_items = list(aligned_faces.items())
    for offset in range(0, len(aligned_items), ARCFACE_BATCH_SIZE):
        batch = aligned_items[offset : offset + ARCFACE_BATCH_SIZE]
        vectors = engine.recognizer.embed_aligned([item[1] for item in batch])
        for (path, _), vector in zip(batch, vectors):
            embeddings[path] = vector
    return embeddings, aligned_faces, detection_times, face_counts


def _select_lfw_target_face(image: np.ndarray, faces):
    image_height, image_width = image.shape[:2]
    center_x = image_width / 2.0
    center_y = image_height / 2.0

    def target_rank(face):
        x1, y1, x2, y2 = (float(value) for value in face.bbox)
        face_center_x = (x1 + x2) / 2.0
        face_center_y = (y1 + y2) / 2.0
        normalized_distance = (
            ((face_center_x - center_x) / max(image_width, 1)) ** 2
            + ((face_center_y - center_y) / max(image_height, 1)) ** 2
        )
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        return normalized_distance, -area

    return min(faces, key=target_rank)


def _verification_scores(
    pairs: Sequence[VerificationPair],
    embeddings: Mapping[str, np.ndarray],
) -> Tuple[List[float], List[float], int]:
    genuine: List[float] = []
    impostor: List[float] = []
    skipped = 0
    for pair in pairs:
        first = embeddings.get(str(pair.first))
        second = embeddings.get(str(pair.second))
        if first is None or second is None:
            skipped += 1
            continue
        score = float(np.clip(np.dot(first, second), -1.0, 1.0))
        (genuine if pair.same_person else impostor).append(score)
    return genuine, impostor, skipped


def _identification_result(
    identities: Sequence[IdentitySplit],
    unknowns: Sequence[Path],
    embeddings: Mapping[str, np.ndarray],
    threshold: float,
) -> Mapping[str, object]:
    valid = [
        identity
        for identity in identities
        if str(identity.enrollment) in embeddings and str(identity.probe) in embeddings
    ]
    if not valid:
        return {
            "requested_identities": len(identities),
            "evaluated_identities": 0,
            "rank1_accuracy": 0.0,
            "thresholded_correct_rate": 0.0,
            "unknowns_evaluated": 0,
            "unknown_rejection_rate": 0.0,
        }

    gallery = normalized_rows(embeddings[str(item.enrollment)] for item in valid)
    probes = normalized_rows(embeddings[str(item.probe)] for item in valid)
    similarities = probes @ gallery.T
    predicted = np.argmax(similarities, axis=1)
    best_scores = similarities[np.arange(len(valid)), predicted]
    expected = np.arange(len(valid))
    rank1_correct = predicted == expected
    thresholded_correct = rank1_correct & (best_scores >= threshold)

    valid_unknowns = [embeddings[str(path)] for path in unknowns if str(path) in embeddings]
    unknown_rejection_rate = 0.0
    if valid_unknowns:
        unknown_matrix = normalized_rows(valid_unknowns)
        unknown_best = np.max(unknown_matrix @ gallery.T, axis=1)
        unknown_rejection_rate = float(np.mean(unknown_best < threshold))

    return {
        "requested_identities": len(identities),
        "evaluated_identities": len(valid),
        "rank1_accuracy": round(float(np.mean(rank1_correct)), 6),
        "thresholded_correct_rate": round(float(np.mean(thresholded_correct)), 6),
        "mean_best_similarity": round(float(np.mean(best_scores)), 6),
        "unknowns_requested": len(unknowns),
        "unknowns_evaluated": len(valid_unknowns),
        "unknown_rejection_rate": round(unknown_rejection_rate, 6),
    }


def _batch_performance(aligned_faces: Sequence[np.ndarray], recognizer, limit: int):
    samples = list(aligned_faces[:limit])
    if not samples:
        return []
    results = []
    for batch_size in (1, 8, 16):
        recognizer.embed_aligned(samples[: min(batch_size, len(samples))])
        started = time.perf_counter()
        processed = 0
        for offset in range(0, len(samples), batch_size):
            batch = samples[offset : offset + batch_size]
            recognizer.embed_aligned(batch)
            processed += len(batch)
        elapsed = time.perf_counter() - started
        results.append(
            {
                "batch_size": batch_size,
                "images": processed,
                "elapsed_ms": round(elapsed * 1000.0, 3),
                "milliseconds_per_image": round(elapsed * 1000.0 / processed, 3),
                "images_per_second": round(processed / max(elapsed, 1e-12), 3),
            }
        )
    return results


def _warnings(result: Mapping[str, object]) -> List[str]:
    warnings: List[str] = []
    detection = result["detection"]
    verification = result["verification"]
    identification = result["identification"]
    if detection["single_face_rate"] < 0.98:
        warnings.append("Yuz tespitinde kacirilan veya coklu bulunan LFW kareleri incelenmeli.")
    if verification["configured"]["accuracy"] + 0.005 < verification["recommended"]["accuracy"]:
        warnings.append("Mevcut esik, benchmark tarafindan bulunan esikten belirgin bicimde daha zayif.")
    if verification["configured"]["false_accept_rate"] > 0.01:
        warnings.append("Yanlis kabul orani %1'in uzerinde; esik ve referans kalitesi incelenmeli.")
    if identification["rank1_accuracy"] < 0.90:
        warnings.append("Kapali kume Rank-1 kimlik bulma basarisi %90'in altinda.")
    if identification["unknowns_evaluated"] and identification["unknown_rejection_rate"] < 0.95:
        warnings.append("Bilinmeyen kisileri reddetme basarisi %95'in altinda.")
    if not warnings:
        warnings.append("Secilen benchmark orneklerinde kritik bir esik ihlali gorulmedi.")
    return warnings


def run(args: argparse.Namespace) -> Mapping[str, object]:
    _validate_paths(args.dataset_root, args.pairs_file)
    if args.full:
        args.max_pairs_per_class = 3000
        args.max_identities = max(args.max_identities, 1000)
        args.max_unknowns = max(args.max_unknowns, 1000)
        args.performance_samples = max(args.performance_samples, 256)
    for name in (
        "max_pairs_per_class",
        "max_identities",
        "max_unknowns",
        "performance_samples",
    ):
        _positive(name, getattr(args, name))

    pairs = load_verification_pairs(
        args.pairs_file,
        args.dataset_root,
        args.max_pairs_per_class,
        args.seed,
    )
    identities, unknowns = collect_identity_splits(
        args.dataset_root,
        args.max_identities,
        args.max_unknowns,
        args.seed,
    )
    image_paths = unique_image_paths(pairs, identities, unknowns)
    missing_paths = [str(path) for path in image_paths if not path.is_file()]
    image_paths = [path for path in image_paths if path.is_file()]

    print("Benchmark modelleri hazirlaniyor...", flush=True)
    engine = get_yolo_arcface_engine()
    print(f"Toplam {len(image_paths)} benzersiz LFW fotografi islenecek.", flush=True)
    started = time.perf_counter()
    embeddings, aligned_faces, detection_times, face_counts = _process_images(
        image_paths,
        engine,
    )
    pipeline_elapsed = time.perf_counter() - started

    genuine, impostor, skipped_pairs = _verification_scores(pairs, embeddings)
    curve = threshold_curve(genuine, impostor)
    if not curve:
        raise RuntimeError("Benchmark icin yeterli gecerli ayni/farkli kisi cifti kalmadi.")
    recommended = best_threshold(curve)
    configured_threshold = float(os.getenv("FACE_MATCH_THRESHOLD", "0.45"))
    configured = min(curve, key=lambda item: abs(item.threshold - configured_threshold))
    identification = _identification_result(
        identities,
        unknowns,
        embeddings,
        configured_threshold,
    )

    benchmark_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    created_at = datetime.now(timezone.utc).isoformat()
    processed_count = len(image_paths) - face_counts["decode_error"]
    warm_detection_times = detection_times[1:] or detection_times
    result: Dict[str, object] = {
        "schema_version": 1,
        "benchmark_id": benchmark_id,
        "created_at": created_at,
        "safety": {
            "database_writes": False,
            "qdrant_writes": False,
            "minio_writes": False,
            "dataset_read_only": True,
        },
        "configuration": {
            "seed": args.seed,
            "configured_threshold": configured_threshold,
            "arcface_batch_size": ARCFACE_BATCH_SIZE,
            "yolo_model_path": YOLO_MODEL_PATH,
            "arcface_model_path": ARCFACE_MODEL_PATH,
            "yolo_model_sha256": _sha256(YOLO_MODEL_PATH),
            "arcface_model_sha256": _sha256(ARCFACE_MODEL_PATH),
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "gpu": _gpu_info(),
            "yolo_providers": engine.detector.providers(),
            "arcface_providers": engine.recognizer.providers(),
        },
        "dataset": {
            "name": "LFW deepfunneled",
            "images_requested": len(image_paths),
            "missing_image_paths": len(missing_paths),
            "verification_pairs_requested": len(pairs),
            "identities_requested": len(identities),
            "unknowns_requested": len(unknowns),
        },
        "detection": {
            "images_processed": processed_count,
            "zero_faces": face_counts["zero"],
            "single_face": face_counts["single"],
            "multiple_faces": face_counts["multiple"],
            "decode_errors": face_counts["decode_error"],
            "single_face_rate": round(face_counts["single"] / max(1, processed_count), 6),
            "latency_ms": {
                "cold_start": round(detection_times[0] if detection_times else 0.0, 3),
                "mean": round(
                    float(np.mean(warm_detection_times)) if warm_detection_times else 0.0,
                    3,
                ),
                "median": round(percentile(warm_detection_times, 50), 3),
                "p95": round(percentile(warm_detection_times, 95), 3),
                "maximum": round(max(warm_detection_times, default=0.0), 3),
            },
        },
        "verification": {
            "genuine_pairs_evaluated": len(genuine),
            "impostor_pairs_evaluated": len(impostor),
            "pairs_skipped": skipped_pairs,
            "roc_auc": round(roc_auc(genuine, impostor), 6),
            "configured": metric_to_dict(configured),
            "recommended": metric_to_dict(recommended),
            "genuine_similarity": {
                "mean": round(float(np.mean(genuine)), 6),
                "p05": round(percentile(genuine, 5), 6),
            },
            "impostor_similarity": {
                "mean": round(float(np.mean(impostor)), 6),
                "p95": round(percentile(impostor, 95), 6),
            },
        },
        "identification": identification,
        "performance": {
            "end_to_end_seconds": round(pipeline_elapsed, 3),
            "end_to_end_images_per_second": round(
                len(image_paths) / max(pipeline_elapsed, 1e-12), 3
            ),
            "arcface_batches": _batch_performance(
                list(aligned_faces.values()),
                engine.recognizer,
                args.performance_samples,
            ),
        },
    }
    result["warnings"] = _warnings(result)
    output_directory = args.output_root / benchmark_id
    report_paths = write_reports(output_directory, result, curve)
    print(json.dumps({"result": result, "reports": report_paths}, ensure_ascii=False, indent=2))
    return {"result": result, "reports": report_paths}


def main() -> int:
    try:
        run(_parser().parse_args())
    except Exception as error:
        print(f"Benchmark basarisiz: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
