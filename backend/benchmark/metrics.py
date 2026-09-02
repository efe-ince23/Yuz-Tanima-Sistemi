from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ThresholdMetric:
    threshold: float
    accuracy: float
    true_accept_rate: float
    false_accept_rate: float
    false_reject_rate: float
    balanced_accuracy: float


def percentile(values: Sequence[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile_value))


def evaluate_threshold(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    threshold: float,
) -> ThresholdMetric:
    genuine_accepts = int(np.count_nonzero(genuine_scores >= threshold))
    impostor_accepts = int(np.count_nonzero(impostor_scores >= threshold))
    genuine_count = max(1, genuine_scores.size)
    impostor_count = max(1, impostor_scores.size)
    true_accept_rate = genuine_accepts / genuine_count
    false_accept_rate = impostor_accepts / impostor_count
    false_reject_rate = 1.0 - true_accept_rate
    true_reject_rate = 1.0 - false_accept_rate
    accuracy = (genuine_accepts + impostor_count - impostor_accepts) / (
        genuine_count + impostor_count
    )
    return ThresholdMetric(
        threshold=float(threshold),
        accuracy=float(accuracy),
        true_accept_rate=float(true_accept_rate),
        false_accept_rate=float(false_accept_rate),
        false_reject_rate=float(false_reject_rate),
        balanced_accuracy=float((true_accept_rate + true_reject_rate) / 2.0),
    )


def threshold_curve(
    genuine_scores: Sequence[float],
    impostor_scores: Sequence[float],
    minimum: float = 0.30,
    maximum: float = 0.70,
    step: float = 0.01,
) -> List[ThresholdMetric]:
    genuine = np.asarray(genuine_scores, dtype=np.float32)
    impostor = np.asarray(impostor_scores, dtype=np.float32)
    if genuine.size == 0 or impostor.size == 0:
        return []
    count = int(round((maximum - minimum) / step)) + 1
    return [
        evaluate_threshold(genuine, impostor, float(threshold))
        for threshold in np.linspace(minimum, maximum, count)
    ]


def best_threshold(metrics: Sequence[ThresholdMetric]) -> ThresholdMetric:
    if not metrics:
        raise ValueError("At least one threshold metric is required.")
    return max(
        metrics,
        key=lambda item: (
            item.balanced_accuracy,
            -item.false_accept_rate,
            -abs(item.threshold - 0.5),
        ),
    )


def roc_auc(genuine_scores: Sequence[float], impostor_scores: Sequence[float]) -> float:
    genuine = np.asarray(genuine_scores, dtype=np.float64)
    impostor = np.asarray(impostor_scores, dtype=np.float64)
    if genuine.size == 0 or impostor.size == 0:
        return 0.0
    wins = 0
    ties = 0
    for offset in range(0, genuine.size, 256):
        differences = genuine[offset : offset + 256, None] - impostor[None, :]
        wins += int(np.count_nonzero(differences > 0))
        ties += int(np.count_nonzero(differences == 0))
    comparison_count = genuine.size * impostor.size
    return float((wins + 0.5 * ties) / comparison_count)


def metric_to_dict(metric: ThresholdMetric) -> Dict[str, float]:
    return {
        "threshold": round(metric.threshold, 4),
        "accuracy": round(metric.accuracy, 6),
        "true_accept_rate": round(metric.true_accept_rate, 6),
        "false_accept_rate": round(metric.false_accept_rate, 6),
        "false_reject_rate": round(metric.false_reject_rate, 6),
        "balanced_accuracy": round(metric.balanced_accuracy, 6),
    }


def normalized_rows(values: Iterable[np.ndarray]) -> np.ndarray:
    matrix = np.asarray(list(values), dtype=np.float32)
    if matrix.size == 0:
        return np.empty((0, 0), dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)
