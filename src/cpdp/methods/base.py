from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from cpdp.types import LabeledProject, PredictionResult, UnlabeledProject


@dataclass(frozen=True)
class MethodProvenance:
    display_name: str
    implementation: str
    fidelity: str
    source_url: str
    source_revision: str
    target_label_policy: str
    notes: str


class BaseMethod(ABC):
    name: str
    provenance: MethodProvenance

    def __init__(self, project_root, params: Mapping[str, Any]) -> None:
        self.project_root = project_root
        self.params = dict(params)

    @abstractmethod
    def fit_predict(
        self,
        sources: Sequence[LabeledProject],
        target: UnlabeledProject,
        seed: int,
    ) -> PredictionResult:
        raise NotImplementedError

