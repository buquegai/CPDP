from __future__ import annotations

from typing import Sequence

from cpdp.methods.base import BaseMethod, MethodProvenance
from cpdp.methods.registry import register_method
from cpdp.types import LabeledProject, PredictionResult, UnlabeledProject


class JEPAMethodTemplate(BaseMethod):
    """Starting point for the paper's JEPA method; not registered by default."""

    name = "jepa"
    provenance = MethodProvenance(
        display_name="JEPA (custom)",
        implementation="User implementation in the unified CPDP framework",
        fidelity="proposed-method",
        source_url="local",
        source_revision="unreleased",
        target_label_policy="Target labels are unavailable until common evaluation",
        notes="Replace fit_predict with the final JEPA training and scoring pipeline.",
    )

    def fit_predict(
        self,
        sources: Sequence[LabeledProject],
        target: UnlabeledProject,
        seed: int,
    ) -> PredictionResult:
        raise NotImplementedError(
            "Implement JEPA training using source labels and target.features only"
        )


def register_methods() -> None:
    register_method(JEPAMethodTemplate.name, JEPAMethodTemplate)
