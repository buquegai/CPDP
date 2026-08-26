from __future__ import annotations

from typing import Any, Mapping, Sequence

from cpdp.methods.base import BaseMethod, MethodProvenance
from cpdp.methods.proposed_pipeline import run_oversampler_then_predictor
from cpdp.oversampling.pipeline import IPGDOOversampler
from cpdp.oversampling_seed_search.cache import load_selected_augmentation
from cpdp.predictors.transformer_jepa import TransformerJEPAPredictor
from cpdp.types import LabeledProject, PredictionResult, UnlabeledProject


class IPGDOTransformerJEPAMethod(BaseMethod):
    name = "ipgdo_transformer_jepa"
    provenance = MethodProvenance(
        display_name="IPGDO + Transformer-JEPA",
        implementation=(
            "Implicit Physics-Guided Diffusion Oversampling and transductive "
            "Transformer-JEPA prediction"
        ),
        fidelity="proposed-method-architecture-variant",
        source_url="local",
        source_revision="unreleased",
        target_label_policy="Target labels are used only by the common evaluator",
        notes=(
            "Controlled architecture variant of the proposed method. Context encoder, "
            "EMA target encoder, and JEPA predictor are Transformers; all other data, "
            "loss, validation, and training contracts remain unchanged."
        ),
    )

    def fit_predict(
        self,
        sources: Sequence[LabeledProject],
        target: UnlabeledProject,
        seed: int,
    ) -> PredictionResult:
        source_keys = {source.key for source in sources}
        if target.key in source_keys:
            raise ValueError("Held-out target project cannot appear among source projects")
        if any(source.features.family != target.features.family for source in sources):
            raise ValueError("Sources and target must belong to the same dataset family")
        oversampling_config = self._section("oversampling")
        oversampler = IPGDOOversampler.from_mapping(oversampling_config)
        predictor = TransformerJEPAPredictor.from_mapping(
            self._section("predictor")
        )
        augmentation, oversampling_seed = load_selected_augmentation(
            self.project_root,
            target,
            sources,
            oversampling_config,
            self._section("oversampling_reuse"),
        )
        return run_oversampler_then_predictor(
            oversampler,
            predictor,
            sources,
            target,
            seed,
            oversampling_seed=oversampling_seed,
            augmentation=augmentation,
        )

    def _section(self, name: str) -> Mapping[str, Any]:
        values = self.params.get(name, {})
        if not isinstance(values, Mapping):
            raise TypeError(f"{self.name}.{name} must be a mapping")
        return values
