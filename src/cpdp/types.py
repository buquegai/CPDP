from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class ProjectFeatures:
    family: str
    name: str
    values: np.ndarray
    feature_names: Tuple[str, ...]
    effort: np.ndarray
    effort_name: str
    row_ids: np.ndarray
    cleaning: Dict[str, int] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.family}/{self.name}"

    @property
    def n_samples(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.values.shape[1])


@dataclass(frozen=True)
class UnlabeledFeatures:
    family: str
    name: str
    values: np.ndarray
    feature_names: Tuple[str, ...]
    effort: np.ndarray
    effort_name: str
    row_ids: np.ndarray
    cleaning: Dict[str, int] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.family}/{self.name}"

    @property
    def n_samples(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.values.shape[1])


@dataclass(frozen=True)
class UnlabeledProject:
    """Target-domain view deliberately containing no defect labels."""

    features: UnlabeledFeatures

    @property
    def key(self) -> str:
        return self.features.key


@dataclass(frozen=True)
class LabeledProject:
    features: ProjectFeatures
    labels: np.ndarray

    @property
    def key(self) -> str:
        return self.features.key

    def without_labels(self) -> UnlabeledProject:
        return UnlabeledProject(
            features=UnlabeledFeatures(
                family=self.features.family,
                name=self.features.name,
                values=self.features.values,
                feature_names=self.features.feature_names,
                effort=self.features.effort,
                effort_name=self.features.effort_name,
                row_ids=self.features.row_ids,
                cleaning=dict(self.features.cleaning),
            )
        )


@dataclass
class PredictionResult:
    scores: np.ndarray
    threshold: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    evaluated_row_ids: Optional[np.ndarray] = None
