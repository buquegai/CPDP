from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

from cpdp.oversampling.legacy_config import ValidityConfig


@dataclass(frozen=True)
class RobustAffineScaler:
    center: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "RobustAffineScaler":
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or len(array) == 0:
            raise ValueError("RobustAffineScaler requires a non-empty 2D array")
        center = np.median(array, axis=0)
        first_quartile, third_quartile = np.quantile(array, [0.25, 0.75], axis=0)
        scale = third_quartile - first_quartile
        standard_deviation = np.std(array, axis=0)
        scale = np.where(scale > 1e-12, scale, standard_deviation)
        scale = np.where(scale > 1e-12, scale, 1.0)
        return cls(center=center.astype(np.float64), scale=scale.astype(np.float64))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float64) - self.center) / self.scale

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float64) * self.scale + self.center

    def inverse_transform_torch(self, values: Any) -> Any:
        import torch

        center = torch.as_tensor(self.center, dtype=values.dtype, device=values.device)
        scale = torch.as_tensor(self.scale, dtype=values.dtype, device=values.device)
        return values * scale + center

    def as_dict(self) -> Dict[str, Any]:
        return {"center": self.center.tolist(), "scale": self.scale.tolist()}


@dataclass(frozen=True)
class NumericValidityFilter:
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    nonnegative: np.ndarray
    tolerance: float

    @classmethod
    def fit(
        cls, values: np.ndarray, config: ValidityConfig
    ) -> "NumericValidityFilter":
        array = np.asarray(values, dtype=np.float64)
        lower = np.quantile(array, config.lower_quantile, axis=0)
        upper = np.quantile(array, config.upper_quantile, axis=0)
        first_quartile, third_quartile = np.quantile(array, [0.25, 0.75], axis=0)
        spread = third_quartile - first_quartile
        fallback = np.std(array, axis=0)
        spread = np.where(spread > 1e-12, spread, fallback)
        spread = np.where(spread > 1e-12, spread, 1.0)
        lower = lower - config.iqr_margin * spread
        upper = upper + config.iqr_margin * spread
        nonnegative = np.min(array, axis=0) >= -config.nonnegative_tolerance
        lower = np.where(nonnegative, np.maximum(lower, 0.0), lower)
        return cls(lower, upper, nonnegative, config.nonnegative_tolerance)

    def evaluate(self, candidates: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
        values = np.asarray(candidates, dtype=np.float64)
        finite = np.isfinite(values).all(axis=1)
        below = (values < self.lower_bounds - self.tolerance).any(axis=1)
        above = (values > self.upper_bounds + self.tolerance).any(axis=1)
        bounded = ~below & ~above
        if np.any(self.nonnegative):
            negative = (
                values[:, self.nonnegative] < -self.tolerance
            ).any(axis=1)
        else:
            negative = np.zeros(len(values), dtype=bool)
        accepted = finite & bounded & ~negative
        return accepted, {
            "candidates": int(len(values)),
            "accepted": int(accepted.sum()),
            "nonfinite_rows": int((~finite).sum()),
            "rows_below_lower_bound": int(below.sum()),
            "rows_above_upper_bound": int(above.sum()),
            "rows_with_negative_nonnegative_metric": int(negative.sum()),
        }

    def mask(self, candidates: np.ndarray) -> np.ndarray:
        return self.evaluate(candidates)[0]
