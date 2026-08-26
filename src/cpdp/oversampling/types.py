from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from cpdp.types import LabeledProject


@dataclass(frozen=True)
class RoleComponent:
    feature_name: str
    feature_index: int
    weight: float
    center: float
    scale: float


@dataclass(frozen=True)
class SemanticRole:
    name: str
    components: Tuple[RoleComponent, ...]


@dataclass(frozen=True)
class RoleMapping:
    roles: Tuple[SemanticRole, ...]

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(role.name for role in self.roles)

    def as_dict(self) -> Dict[str, Any]:
        return {
            role.name: [
                {
                    "feature": component.feature_name,
                    "weight": component.weight,
                    "center": component.center,
                    "scale": component.scale,
                }
                for component in role.components
            ]
            for role in self.roles
        }


@dataclass(frozen=True)
class FormulaTerm:
    name: str
    operation: str
    left_role: str
    right_role: str = ""


@dataclass(frozen=True)
class FormulaBank:
    term_names: Tuple[str, ...]
    term_centers: np.ndarray
    term_scales: np.ndarray
    global_coefficients: np.ndarray
    global_intercept: float
    project_coefficients: np.ndarray
    project_intercepts: np.ndarray
    project_names: Tuple[str, ...]
    reliability: float
    validation_auc: float
    enabled: bool

    def as_dict(self) -> Dict[str, Any]:
        selected = [
            {
                "term": name,
                "coefficient": float(coefficient),
            }
            for name, coefficient in zip(self.term_names, self.global_coefficients)
            if abs(float(coefficient)) > 0.0
        ]
        return {
            "enabled": self.enabled,
            "reliability": self.reliability,
            "project_refit_held_out_auc": self.validation_auc,
            "global_intercept": self.global_intercept,
            "selected_terms": selected,
            "project_specific_coefficients": {
                project: {
                    "intercept": float(self.project_intercepts[index]),
                    "coefficients": self.project_coefficients[index].tolist(),
                }
                for index, project in enumerate(self.project_names)
            },
        }


@dataclass(frozen=True)
class ProjectGenerationDiagnostics:
    project: str
    clean_samples: int
    real_defects: int
    requested_synthetic: int
    generation_rounds: int
    generated_candidates: int
    selected_synthetic: int
    formula_guidance: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AugmentationResult:
    projects: Tuple[LabeledProject, ...]
    metadata: Mapping[str, Any]


def freeze_array(values: Sequence[Any], dtype: Any = None) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    array.setflags(write=False)
    return array
