from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from sklearn.neighbors import NearestNeighbors

from cpdp.oversampling.config import KNNFilterConfig
from cpdp.oversampling.preprocessing import RobustAffineScaler


@dataclass(frozen=True)
class KNNFilterEvaluation:
    accepted: np.ndarray
    quality_scores: np.ndarray
    mean_distances: np.ndarray
    metadata: Dict[str, Any]


class DefectManifoldFilter:
    """Keep candidates inside the normal KNN-distance range of real defects."""

    def __init__(
        self, real_defects: np.ndarray, config: KNNFilterConfig
    ) -> None:
        defects = np.asarray(real_defects, dtype=np.float64)
        if defects.ndim != 2 or len(defects) < 2:
            raise ValueError("KNN filtering requires at least two real defect samples")
        if not np.isfinite(defects).all():
            raise ValueError("Real defect samples must contain only finite values")

        self.config = config
        self.scaler = RobustAffineScaler.fit(defects)
        self.normalized_defects = self.scaler.transform(defects)
        self.neighbors = min(config.neighbors, len(defects) - 1)

        internal_model = NearestNeighbors(
            n_neighbors=self.neighbors + 1, metric="euclidean"
        )
        internal_model.fit(self.normalized_defects)
        internal_distances = internal_model.kneighbors(
            self.normalized_defects, return_distance=True
        )[0][:, 1:]
        internal_mean = np.mean(internal_distances, axis=1)
        self.lower_distance = float(
            np.quantile(internal_mean, config.lower_quantile)
        )
        self.upper_distance = float(
            np.quantile(internal_mean, config.upper_quantile)
        )
        self.target_distance = float(np.median(internal_mean))
        self.distance_scale = max(
            self.upper_distance - self.lower_distance,
            abs(self.target_distance),
            1e-12,
        )
        self.candidate_model = NearestNeighbors(
            n_neighbors=self.neighbors, metric="euclidean"
        )
        self.candidate_model.fit(self.normalized_defects)

    def evaluate(self, candidates: np.ndarray) -> KNNFilterEvaluation:
        values = np.asarray(candidates, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.normalized_defects.shape[1]:
            raise ValueError("Candidates have an incompatible shape")
        if len(values) == 0:
            empty_float = np.empty(0, dtype=np.float64)
            return KNNFilterEvaluation(
                accepted=np.zeros(0, dtype=bool),
                quality_scores=empty_float,
                mean_distances=empty_float,
                metadata={"candidates": 0, "accepted": 0, "nonfinite": 0},
            )

        finite = np.isfinite(values).all(axis=1)
        mean_distances = np.full(len(values), np.inf, dtype=np.float64)
        if finite.any():
            normalized = self.scaler.transform(values[finite])
            distances = self.candidate_model.kneighbors(
                normalized, return_distance=True
            )[0]
            mean_distances[finite] = np.mean(distances, axis=1)

        accepted = (
            finite
            & (mean_distances >= self.lower_distance)
            & (mean_distances <= self.upper_distance)
        )
        quality_scores = (
            np.abs(mean_distances - self.target_distance) / self.distance_scale
        )
        finite_distances = mean_distances[finite]
        metadata: Dict[str, Any] = {
            "candidates": int(len(values)),
            "accepted": int(accepted.sum()),
            "nonfinite": int((~finite).sum()),
        }
        if len(finite_distances):
            metadata.update(
                {
                    "candidate_distance_min": float(np.min(finite_distances)),
                    "candidate_distance_median": float(np.median(finite_distances)),
                    "candidate_distance_max": float(np.max(finite_distances)),
                }
            )
        return KNNFilterEvaluation(
            accepted=accepted,
            quality_scores=quality_scores,
            mean_distances=mean_distances,
            metadata=metadata,
        )

    def metadata(self) -> Dict[str, Any]:
        return {
            "neighbors": int(self.neighbors),
            "real_defects": int(len(self.normalized_defects)),
            "internal_distance_lower": self.lower_distance,
            "internal_distance_median": self.target_distance,
            "internal_distance_upper": self.upper_distance,
            "quality_rule": (
                "candidate mean KNN distance must remain inside the real-defect "
                "internal-distance quantile interval"
            ),
            "ranking_rule": (
                "smaller absolute distance from the real-defect internal median is better"
            ),
        }
