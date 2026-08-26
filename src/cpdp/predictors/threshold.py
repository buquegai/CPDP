from __future__ import annotations

import numpy as np
from sklearn.metrics import matthews_corrcoef


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
