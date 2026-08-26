from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping

from cpdp.predictors.kan_jepa import (
    JEPAPredictorBase,
    LossWeights,
    OptimizationConfig,
    ViewConfig,
)
from cpdp.predictors.transformer import (
    TransformerEncoderConfig,
    TransformerPredictorConfig,
    build_transformer_jepa_model,
)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


@dataclass(frozen=True)
class TransformerJEPAConfig:
    device: str = "auto"
    input_clip: float = 8.0
    data_flow_mode: str = "split"
    encoder: TransformerEncoderConfig = field(
        default_factory=TransformerEncoderConfig
    )
    jepa_predictor: TransformerPredictorConfig = field(
        default_factory=TransformerPredictorConfig
    )
    views: ViewConfig = field(default_factory=ViewConfig)
    losses: LossWeights = field(default_factory=LossWeights)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "TransformerJEPAConfig":
        config = cls(
            device=str(values.get("device", "auto")),
            input_clip=float(values.get("input_clip", 8.0)),
            data_flow_mode=str(values.get("data_flow_mode", "split")).strip().lower(),
            encoder=TransformerEncoderConfig.from_mapping(
                _mapping(values.get("encoder"), "encoder")
            ),
            jepa_predictor=TransformerPredictorConfig.from_mapping(
                _mapping(values.get("jepa_predictor"), "jepa_predictor")
            ),
            views=ViewConfig.from_mapping(_mapping(values.get("views"), "views")),
            losses=LossWeights.from_mapping(_mapping(values.get("losses"), "losses")),
            optimization=OptimizationConfig.from_mapping(
                _mapping(values.get("optimization"), "optimization")
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("predictor.device must be 'auto', 'cpu', or 'cuda'")
        if self.input_clip <= 0:
            raise ValueError("predictor.input_clip must be positive")
        if self.data_flow_mode not in {"split", "shared_augmented_source"}:
            raise ValueError(
                "predictor.data_flow_mode must be 'split' or " 
                "'shared_augmented_source'"
            )


class TransformerJEPAPredictor(JEPAPredictorBase):
    """Transformer-JEPA using the established CPDP training protocol."""

    model_name = "Transformer-JEPA"
    metadata_name = "transformer-jepa"
    progress_name = "transformer_jepa"

    def __init__(self, config: TransformerJEPAConfig) -> None:
        super().__init__(config)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "TransformerJEPAPredictor":
        return cls(TransformerJEPAConfig.from_mapping(values))

    def _build_model(self, torch, feature_dim: int):
        return build_transformer_jepa_model(
            torch,
            feature_dim=feature_dim,
            encoder_config=self.config.encoder,
            predictor_config=self.config.jepa_predictor,
        )

    @staticmethod
    def _architecture_metadata() -> Dict[str, str]:
        return {
            "context_encoder": "Transformer",
            "target_encoder": "EMA-Transformer",
            "latent_predictor": "Transformer",
        }
