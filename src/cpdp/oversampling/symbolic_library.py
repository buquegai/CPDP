from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations
from typing import Dict, List, Tuple

import numpy as np

from cpdp.oversampling.semantic_roles import (
    transform_roles_numpy,
    transform_roles_torch,
)
from cpdp.oversampling.types import FormulaTerm, RoleMapping


def _signed_log_numpy(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.log1p(np.abs(values))


def _evaluate_numpy(
    term: FormulaTerm, roles: Dict[str, np.ndarray]
) -> np.ndarray:
    left = roles[term.left_role]
    if term.operation == "identity":
        return left
    if term.operation == "signed_log":
        return _signed_log_numpy(left)
    if term.operation == "soft_square":
        return left * np.abs(left)
    right = roles[term.right_role]
    if term.operation == "product":
        return left * right
    if term.operation == "safe_log_ratio":
        return _signed_log_numpy(left) / (1.0 + np.abs(_signed_log_numpy(right)))
    if term.operation == "cross_log":
        return left * _signed_log_numpy(right)
    raise KeyError(f"Unknown symbolic operation: {term.operation}")


def _evaluate_torch(term: FormulaTerm, roles):
    import torch

    left = roles[term.left_role]
    signed_log = lambda value: torch.sign(value) * torch.log1p(torch.abs(value))
    if term.operation == "identity":
        return left
    if term.operation == "signed_log":
        return signed_log(left)
    if term.operation == "soft_square":
        return left * torch.abs(left)
    right = roles[term.right_role]
    if term.operation == "product":
        return left * right
    if term.operation == "safe_log_ratio":
        return signed_log(left) / (1.0 + torch.abs(signed_log(right)))
    if term.operation == "cross_log":
        return left * signed_log(right)
    raise KeyError(f"Unknown symbolic operation: {term.operation}")


def build_dynamic_terms(role_names: Tuple[str, ...]) -> Tuple[FormulaTerm, ...]:
    terms: List[FormulaTerm] = []
    for role in role_names:
        terms.extend(
            (
                FormulaTerm(role, "identity", role),
                FormulaTerm(f"slog({role})", "signed_log", role),
                FormulaTerm(f"soft_square({role})", "soft_square", role),
            )
        )
    for left, right in combinations(role_names, 2):
        terms.append(FormulaTerm(f"{left}*{right}", "product", left, right))
    for left, right in permutations(role_names, 2):
        terms.append(
            FormulaTerm(
                f"safe_log_ratio({left},{right})",
                "safe_log_ratio",
                left,
                right,
            )
        )
        terms.append(
            FormulaTerm(f"{left}*slog({right})", "cross_log", left, right)
        )
    return tuple(terms)


@dataclass(frozen=True)
class FittedSymbolicLibrary:
    role_mapping: RoleMapping
    terms: Tuple[FormulaTerm, ...]
    centers: np.ndarray
    scales: np.ndarray

    @classmethod
    def fit(
        cls, values: np.ndarray, role_mapping: RoleMapping
    ) -> "FittedSymbolicLibrary":
        terms = build_dynamic_terms(role_mapping.names)
        roles = transform_roles_numpy(values, role_mapping)
        raw = np.column_stack([_evaluate_numpy(term, roles) for term in terms])
        centers = np.median(raw, axis=0)
        lower, upper = np.quantile(raw, [0.25, 0.75], axis=0)
        scales = upper - lower
        fallback = np.std(raw, axis=0)
        scales = np.where(scales > 1e-12, scales, fallback)
        scales = np.where(scales > 1e-12, scales, 1.0)
        return cls(role_mapping, terms, centers, scales)

    @property
    def term_names(self) -> Tuple[str, ...]:
        return tuple(term.name for term in self.terms)

    def transform_numpy(self, values: np.ndarray) -> np.ndarray:
        roles = transform_roles_numpy(values, self.role_mapping)
        raw = np.column_stack([_evaluate_numpy(term, roles) for term in self.terms])
        normalized = (raw - self.centers) / self.scales
        return np.nan_to_num(normalized, nan=0.0, posinf=20.0, neginf=-20.0)

    def transform_torch(self, values):
        import torch

        roles = transform_roles_torch(values, self.role_mapping)
        raw = torch.stack([_evaluate_torch(term, roles) for term in self.terms], dim=1)
        centers = torch.tensor(self.centers, dtype=values.dtype, device=values.device)
        scales = torch.tensor(self.scales, dtype=values.dtype, device=values.device)
        return torch.nan_to_num(
            (raw - centers) / scales, nan=0.0, posinf=20.0, neginf=-20.0
        )
