from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from cpdp.oversampling.legacy_config import FormulaConfig, RoleConfig


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _probability(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


def _reject_unknown(values: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        joined = ", ".join(unknown)
        raise ValueError(f"Unsupported {name} option(s): {joined}")


@dataclass(frozen=True)
class DiffusionConfig:
    hidden_dim: int = 256
    residual_blocks: int = 4
    dropout: float = 0.1
    timesteps: int = 200
    schedule: str = "cosine"
    epochs: int = 300
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    sample_batch_size: int = 1024
    use_amp: bool = True
    normalized_clip: float = 8.0
    physics_warmup_fraction: float = 0.2
    formula_consistency_weight: float = 0.0
    formula_semantic_weight: float = 0.05

    def validate(self) -> None:
        if self.hidden_dim < 16:
            raise ValueError("diffusion.hidden_dim must be at least 16")
        if self.residual_blocks < 1:
            raise ValueError("diffusion.residual_blocks must be at least 1")
        _probability(self.dropout, "diffusion.dropout")
        if self.timesteps < 2:
            raise ValueError("diffusion.timesteps must be at least 2")
        if self.schedule not in {"linear", "cosine"}:
            raise ValueError("diffusion.schedule must be 'linear' or 'cosine'")
        if self.epochs < 1:
            raise ValueError("diffusion.epochs must be at least 1")
        if self.batch_size < 2:
            raise ValueError("diffusion.batch_size must be at least 2")
        _positive(self.learning_rate, "diffusion.learning_rate")
        if self.weight_decay < 0:
            raise ValueError("diffusion.weight_decay cannot be negative")
        _positive(self.gradient_clip_norm, "diffusion.gradient_clip_norm")
        if self.sample_batch_size < 1:
            raise ValueError("diffusion.sample_batch_size must be at least 1")
        if not isinstance(self.use_amp, bool):
            raise TypeError("diffusion.use_amp must be boolean")
        _positive(self.normalized_clip, "diffusion.normalized_clip")
        _probability(
            self.physics_warmup_fraction,
            "diffusion.physics_warmup_fraction",
        )
        if self.formula_consistency_weight < 0:
            raise ValueError(
                "diffusion.formula_consistency_weight cannot be negative"
            )
        if self.formula_semantic_weight < 0:
            raise ValueError(
                "diffusion.formula_semantic_weight cannot be negative"
            )


@dataclass(frozen=True)
class KNNFilterConfig:
    neighbors: int = 5
    lower_quantile: float = 0.10
    upper_quantile: float = 0.90

    def validate(self) -> None:
        if self.neighbors < 1:
            raise ValueError("knn.neighbors must be at least 1")
        _probability(self.lower_quantile, "knn.lower_quantile")
        _probability(self.upper_quantile, "knn.upper_quantile")
        if self.lower_quantile >= self.upper_quantile:
            raise ValueError("knn quantiles must be increasing")


@dataclass(frozen=True)
class OversamplingConfig:
    device: str = "auto"
    target_minority_ratio: float = 1.0
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    role: RoleConfig = field(default_factory=RoleConfig)
    formula: FormulaConfig = field(default_factory=FormulaConfig)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "OversamplingConfig":
        _reject_unknown(
            values,
            {
                "device",
                "target_minority_ratio",
                "diffusion",
                "role",
                "formula",
            },
            "oversampling",
        )
        diffusion_values = _mapping(values.get("diffusion"), "diffusion")
        role_values = _mapping(values.get("role"), "role")
        formula_values = _mapping(values.get("formula"), "formula")
        _reject_unknown(
            diffusion_values,
            set(DiffusionConfig.__dataclass_fields__),
            "diffusion",
        )
        _reject_unknown(role_values, set(RoleConfig.__dataclass_fields__), "role")
        _reject_unknown(
            formula_values,
            set(FormulaConfig.__dataclass_fields__),
            "formula",
        )
        config = cls(
            device=str(values.get("device", "auto")),
            target_minority_ratio=float(values.get("target_minority_ratio", 1.0)),
            diffusion=DiffusionConfig(**diffusion_values),
            role=RoleConfig(**role_values),
            formula=FormulaConfig(**formula_values),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("oversampler.device must be 'auto', 'cpu', or 'cuda'")
        _positive(self.target_minority_ratio, "target_minority_ratio")
        self.diffusion.validate()
        self.role.validate()
        self.formula.validate()
