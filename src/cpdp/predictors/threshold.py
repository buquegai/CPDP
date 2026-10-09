from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef


def select_mcc_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """Select a classification threshold using source-only MCC."""
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(labels) != len(scores):
        raise ValueError("labels and scores must contain the same number of rows")
    if len(np.unique(labels)) < 2:
        return 0.5
    quantiles = np.linspace(0.02, 0.98, 97)
    candidates = np.unique(
        np.concatenate((np.quantile(scores, quantiles), np.asarray([0.5])))
    )
    ranked = []
    for threshold in candidates:
        predictions = (scores >= threshold).astype(np.int64)
        ranked.append(
            (
                matthews_corrcoef(labels, predictions),
                -abs(float(threshold) - 0.5),
                float(threshold),
            )
        )
    return float(max(ranked)[2])


def select_metric_threshold(
    labels: np.ndarray, scores: np.ndarray, metric: str
) -> float:
    """Select a source-only threshold for MCC, F1, G-mean, or fixed 0.5."""
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(labels) != len(scores):
        raise ValueError("labels and scores must contain the same number of rows")
    metric = str(metric).strip().lower()
    if metric in {"mcc", "source-validation-mcc"}:
        return select_mcc_threshold(labels, scores)
    if metric in {"fixed", "fixed_0.5", "0.5"}:
        return 0.5
    if len(np.unique(labels)) < 2:
        return 0.5
    quantiles = np.linspace(0.02, 0.98, 97)
    candidates = np.unique(
        np.concatenate((np.quantile(scores, quantiles), np.asarray([0.5])))
    )
    ranked = []
    for threshold in candidates:
        predictions = (scores >= threshold).astype(np.int64)
        if metric in {"f1", "source-validation-f1"}:
            value = f1_score(labels, predictions, zero_division=0)
        elif metric in {"gmean", "g-mean", "source-validation-gmean"}:
            tp = float(np.sum((labels == 1) & (predictions == 1)))
            fn = float(np.sum((labels == 1) & (predictions == 0)))
            tn = float(np.sum((labels == 0) & (predictions == 0)))
            fp = float(np.sum((labels == 0) & (predictions == 1)))
            recall = tp / (tp + fn) if tp + fn else 0.0
            specificity = tn / (tn + fp) if tn + fp else 0.0
            value = float(np.sqrt(recall * specificity))
        else:
            raise ValueError(f"Unsupported threshold metric: {metric}")
        ranked.append((float(value), -abs(float(threshold) - 0.5), float(threshold)))
    return float(max(ranked)[2])
