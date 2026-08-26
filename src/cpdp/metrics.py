from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


METRIC_NAMES = ("auc", "auprc", "mcc", "f1", "gmean", "popt20")


def _area_to_effort_fraction(
    labels: np.ndarray, effort: np.ndarray, order: np.ndarray, fraction: float
) -> float:
    total_effort = float(np.sum(effort))
    total_defects = float(np.sum(labels))
    if total_effort <= 0 or total_defects <= 0:
        return float("nan")

    effort_limit = fraction * total_effort
    used_effort = 0.0
    found_defects = 0.0
    previous_x = 0.0
    previous_y = 0.0
    area = 0.0

    for index in order:
        item_effort = float(effort[index])
        item_defects = float(labels[index])
        remaining = effort_limit - used_effort
        if remaining <= 0:
            break
        proportion = 1.0 if item_effort <= remaining or item_effort == 0 else remaining / item_effort
        used_effort += item_effort * proportion
        found_defects += item_defects * proportion
        current_x = used_effort / total_effort
        current_y = found_defects / total_defects
        area += (current_x - previous_x) * (current_y + previous_y) / 2.0
        previous_x = current_x
        previous_y = current_y
        if proportion < 1.0:
            break

    if previous_x < fraction:
        area += (fraction - previous_x) * previous_y
    return area


def popt_at_20(labels: np.ndarray, scores: np.ndarray, effort: np.ndarray) -> float:
    safe_effort = np.maximum(np.asarray(effort, dtype=np.float64), 1e-12)
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)

    predicted_order = np.lexsort((safe_effort, -(scores / safe_effort)))
    optimal_order = np.lexsort((safe_effort, -(labels / safe_effort)))
    worst_order = np.lexsort((-safe_effort, labels / safe_effort))

    model_area = _area_to_effort_fraction(labels, safe_effort, predicted_order, 0.20)
    optimal_area = _area_to_effort_fraction(labels, safe_effort, optimal_order, 0.20)
    worst_area = _area_to_effort_fraction(labels, safe_effort, worst_order, 0.20)
    denominator = optimal_area - worst_area
    if not np.isfinite(denominator) or abs(denominator) < 1e-12:
        return float("nan")
    return float(1.0 - (optimal_area - model_area) / denominator)


def evaluate_predictions(
    labels: np.ndarray, scores: np.ndarray, effort: np.ndarray, threshold: float
) -> Dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    effort = np.asarray(effort, dtype=np.float64).reshape(-1)
    if not (len(labels) == len(scores) == len(effort)):
        raise ValueError("Labels, scores, and effort must have equal lengths")
    if not np.isfinite(scores).all():
        raise ValueError("Prediction scores contain non-finite values")

    predictions = (scores >= float(threshold)).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0

    return {
        "auc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "gmean": float(np.sqrt(sensitivity * specificity)),
        "popt20": popt_at_20(labels, scores, effort),
    }

