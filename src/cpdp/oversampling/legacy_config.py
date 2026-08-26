from __future__ import annotations

from dataclasses import dataclass


def _positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _probability(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


@dataclass(frozen=True)
class RoleConfig:
    """Configuration retained only for historical formula experiments."""

    max_components: int = 4
    l1_c: float = 0.2
    minimum_component_weight: float = 0.01

    def validate(self) -> None:
        if self.max_components < 1:
            raise ValueError("role.max_components must be at least 1")
        _positive(self.l1_c, "role.l1_c")
        _probability(self.minimum_component_weight, "role.minimum_component_weight")


@dataclass(frozen=True)
class FormulaConfig:
    """Configuration retained only for historical formula experiments."""

    l1_c: float = 0.1
    max_terms: int = 10
    minimum_reliability: float = 0.0
    use_reliability_weighting: bool = False
    project_blend_strength: float = 50.0
    minimum_class_samples: int = 4

    def validate(self) -> None:
        _positive(self.l1_c, "formula.l1_c")
        if self.max_terms < 1:
            raise ValueError("formula.max_terms must be at least 1")
        _probability(self.minimum_reliability, "formula.minimum_reliability")
        if not isinstance(self.use_reliability_weighting, bool):
            raise TypeError("formula.use_reliability_weighting must be boolean")
        _positive(self.project_blend_strength, "formula.project_blend_strength")
        if self.minimum_class_samples < 2:
            raise ValueError("formula.minimum_class_samples must be at least 2")


@dataclass(frozen=True)
class ValidityConfig:
    """Configuration retained only for the historical numeric filter."""

    lower_quantile: float = 0.001
    upper_quantile: float = 0.999
    iqr_margin: float = 1.5
    nonnegative_tolerance: float = 1e-6

    def validate(self) -> None:
        _probability(self.lower_quantile, "validity.lower_quantile")
        _probability(self.upper_quantile, "validity.upper_quantile")
        if self.lower_quantile >= self.upper_quantile:
            raise ValueError("validity quantiles must be increasing")
        if self.iqr_margin < 0 or self.nonnegative_tolerance < 0:
            raise ValueError("validity margins cannot be negative")
