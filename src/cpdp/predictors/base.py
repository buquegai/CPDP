from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from cpdp.types import LabeledProject, PredictionResult, UnlabeledProject


class BasePredictor(ABC):
    """Prediction interface independent of any oversampling implementation."""

    @abstractmethod
    def fit_predict(
        self,
        sources: Sequence[LabeledProject],
        target: UnlabeledProject,
        seed: int,
    ) -> PredictionResult:
        raise NotImplementedError
