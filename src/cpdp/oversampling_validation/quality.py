from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.stats import wasserstein_distance
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from cpdp.oversampling_validation.config import QualityConfig
from cpdp.types import LabeledProject


QUALITY_METRICS = (
    "normalized_wasserstein_mean",
    "rbf_mmd",
    "correlation_frobenius",
    "exact_duplicate_rate",
    "near_duplicate_rate",
    "defect_knn_distance_mean",
    "defect_knn_distance_ratio",
    "defect_manifold_in_range_rate",
    "heldout_defect_coverage",
    "defect_neighborhood_purity",
)


@dataclass(frozen=True)
class QualityResult:
    metrics: Dict[str, Optional[float]]
    metadata: Dict[str, Any]


def _empty_metrics() -> Dict[str, Optional[float]]:
    return {name: None for name in QUALITY_METRICS}


def _sample(values: np.ndarray, maximum: int, random: np.random.Generator) -> np.ndarray:
    if len(values) <= maximum:
        return values
    return values[random.choice(len(values), size=maximum, replace=False)]


def _correlation(values: np.ndarray) -> Optional[np.ndarray]:
    if len(values) < 2:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = np.corrcoef(values, rowvar=False)
    if np.ndim(correlation) == 0:
        correlation = np.asarray([[float(correlation)]])
    return np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)


def _rbf_mmd(left: np.ndarray, right: np.ndarray) -> Optional[float]:
    if not len(left) or not len(right):
        return None
    pooled = np.vstack((left, right))
    squared = pdist(pooled, metric="sqeuclidean")
    positive = squared[squared > 1e-12]
    gamma = 1.0 if not len(positive) else 1.0 / (2.0 * float(np.median(positive)))
    left_left = np.exp(-gamma * cdist(left, left, metric="sqeuclidean"))
    right_right = np.exp(-gamma * cdist(right, right, metric="sqeuclidean"))
    left_right = np.exp(-gamma * cdist(left, right, metric="sqeuclidean"))
    value = float(left_left.mean() + right_right.mean() - 2.0 * left_right.mean())
    return max(0.0, value)


def evaluate_synthetic_quality(
    train_project: LabeledProject,
    heldout_project: LabeledProject,
    augmented_project: LabeledProject,
    config: QualityConfig,
    seed: int,
) -> QualityResult:
    synthetic_mask = augmented_project.features.row_ids < 0
    synthetic = augmented_project.features.values[synthetic_mask]
    train_values = train_project.features.values
    train_labels = train_project.labels
    train_defects = train_values[train_labels == 1]
    heldout_defects = heldout_project.features.values[heldout_project.labels == 1]
    metrics = _empty_metrics()
    metadata: Dict[str, Any] = {
        "synthetic_samples": int(len(synthetic)),
        "train_real_samples": int(len(train_values)),
        "train_real_defects": int(len(train_defects)),
        "heldout_real_defects": int(len(heldout_defects)),
    }
    if not len(synthetic):
        metadata["reason"] = "no_synthetic_samples"
        return QualityResult(metrics=metrics, metadata=metadata)
    if not len(train_defects):
        metadata["reason"] = "no_training_defects"
        return QualityResult(metrics=metrics, metadata=metadata)

    scaler = StandardScaler().fit(train_values)
    normalized_train = scaler.transform(train_values)
    normalized_defects = scaler.transform(train_defects)
    normalized_synthetic = scaler.transform(synthetic)
    random = np.random.default_rng(seed)
    sampled_synthetic = _sample(normalized_synthetic, config.max_samples, random)
    sampled_heldout = _sample(
        scaler.transform(heldout_defects), config.max_samples, random
    ) if len(heldout_defects) else np.empty((0, train_project.features.n_features))

    exact_distances = cdist(
        normalized_synthetic, normalized_defects, metric="chebyshev"
    ).min(axis=1)
    metrics["exact_duplicate_rate"] = float(
        np.mean(exact_distances <= config.exact_tolerance)
    )

    neighbors = min(config.knn_neighbors, max(1, len(normalized_defects) - 1))
    if len(normalized_defects) >= 2:
        internal_model = NearestNeighbors(n_neighbors=neighbors + 1).fit(
            normalized_defects
        )
        internal_distances = internal_model.kneighbors(
            normalized_defects, return_distance=True
        )[0][:, 1:]
        internal_mean = internal_distances.mean(axis=1)
        nearest_internal = internal_distances[:, 0]
        near_threshold = float(
            np.quantile(nearest_internal, config.near_duplicate_quantile)
        )
        lower = float(
            np.quantile(internal_mean, config.manifold_lower_quantile)
        )
        upper = float(
            np.quantile(internal_mean, config.manifold_upper_quantile)
        )
        reference_mean = float(np.median(internal_mean))
        synthetic_distances = NearestNeighbors(n_neighbors=neighbors).fit(
            normalized_defects
        ).kneighbors(normalized_synthetic, return_distance=True)[0]
        synthetic_mean = synthetic_distances.mean(axis=1)
        metrics["near_duplicate_rate"] = float(
            np.mean(synthetic_distances[:, 0] <= near_threshold)
        )
        metrics["defect_knn_distance_mean"] = float(np.mean(synthetic_mean))
        metrics["defect_knn_distance_ratio"] = (
            None if reference_mean <= 1e-12 else float(np.mean(synthetic_mean) / reference_mean)
        )
        metrics["defect_manifold_in_range_rate"] = float(
            np.mean((synthetic_mean >= lower) & (synthetic_mean <= upper))
        )
        metadata["near_duplicate_distance_threshold"] = near_threshold
        metadata["defect_manifold_distance_lower"] = lower
        metadata["defect_manifold_distance_upper"] = upper
    else:
        metadata["knn_reason"] = "fewer_than_two_training_defects"

    purity_neighbors = min(config.knn_neighbors, len(normalized_train))
    purity_indices = NearestNeighbors(n_neighbors=purity_neighbors).fit(
        normalized_train
    ).kneighbors(normalized_synthetic, return_distance=False)
    metrics["defect_neighborhood_purity"] = float(
        np.mean(train_labels[purity_indices])
    )

    if len(sampled_heldout):
        metrics["normalized_wasserstein_mean"] = float(
            np.mean(
                [
                    wasserstein_distance(
                        sampled_synthetic[:, index], sampled_heldout[:, index]
                    )
                    for index in range(sampled_synthetic.shape[1])
                ]
            )
        )
        metrics["rbf_mmd"] = _rbf_mmd(sampled_synthetic, sampled_heldout)
        synthetic_correlation = _correlation(sampled_synthetic)
        heldout_correlation = _correlation(sampled_heldout)
        if synthetic_correlation is not None and heldout_correlation is not None:
            metrics["correlation_frobenius"] = float(
                np.linalg.norm(synthetic_correlation - heldout_correlation, ord="fro")
                / synthetic_correlation.shape[0]
            )
        heldout_to_train = NearestNeighbors(n_neighbors=1).fit(
            normalized_defects
        ).kneighbors(sampled_heldout, return_distance=True)[0][:, 0]
        coverage_radius = float(
            np.quantile(heldout_to_train, config.coverage_quantile)
        )
        heldout_to_synthetic = NearestNeighbors(n_neighbors=1).fit(
            normalized_synthetic
        ).kneighbors(sampled_heldout, return_distance=True)[0][:, 0]
        metrics["heldout_defect_coverage"] = float(
            np.mean(heldout_to_synthetic <= coverage_radius)
        )
        metadata["heldout_coverage_radius"] = coverage_radius
    else:
        metadata["heldout_distribution_reason"] = "no_heldout_defects"
    return QualityResult(metrics=metrics, metadata=metadata)
