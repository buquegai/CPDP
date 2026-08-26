from __future__ import annotations

from typing import Optional, Sequence

from cpdp.oversampling.types import AugmentationResult
from cpdp.predictors.base import BasePredictor
from cpdp.types import LabeledProject, PredictionResult, UnlabeledProject


def run_oversampler_then_predictor(
    oversampler,
    predictor: BasePredictor,
    sources: Sequence[LabeledProject],
    target: UnlabeledProject,
    seed: int,
    *,
    oversampling_seed: Optional[int] = None,
    augmentation: Optional[AugmentationResult] = None,
) -> PredictionResult:
    selected_oversampling_seed = int(
        seed if oversampling_seed is None else oversampling_seed
    )
    cache_reused = augmentation is not None
    if augmentation is None:
        augmentation = oversampler.fit_resample(sources, selected_oversampling_seed)
    prediction = predictor.fit_predict(augmentation.projects, target, seed)
    return PredictionResult(
        scores=prediction.scores,
        threshold=prediction.threshold,
        metadata={
            "oversampling": augmentation.metadata,
            "predictor": dict(prediction.metadata),
            "random_seeds": {
                "oversampling_seed": selected_oversampling_seed,
                "predictor_seed": int(seed),
            },
            "oversampling_cache_reused": cache_reused,
        },
        evaluated_row_ids=prediction.evaluated_row_ids,
    )
