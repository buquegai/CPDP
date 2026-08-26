from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

from cpdp.oversampling.legacy_config import RoleConfig
from cpdp.oversampling.types import RoleComponent, RoleMapping, SemanticRole


_ROLE_ALIASES = {
    "S": ("loctotal", "loc", "countlinecode", "numberoflinesofcode"),
    "C": ("cyclomaticcomplexity", "wmc", "sumcyclomatic", "avgcyclomatic"),
    "K": ("callpairs", "cbo", "fanout", "fanin", "rfc"),
    "E": ("essentialcomplexity", "sumessential", "avgessential"),
}


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _candidate_priority(feature_name: str, aliases: Sequence[str]) -> int:
    normalized = _normalize(feature_name)
    priorities = []
    for alias in aliases:
        if normalized == alias:
            priorities.append(3)
        elif normalized.endswith(alias):
            priorities.append(2)
    return max(priorities, default=0)


def _robust_columns(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    transformed = np.log1p(np.maximum(values, 0.0))
    centers = np.median(transformed, axis=0)
    lower, upper = np.quantile(transformed, [0.25, 0.75], axis=0)
    scales = upper - lower
    fallback = np.std(transformed, axis=0)
    scales = np.where(scales > 1e-12, scales, fallback)
    scales = np.where(scales > 1e-12, scales, 1.0)
    return (transformed - centers) / scales, centers, scales


class SemanticRoleResolver:
    """Resolve physical roles without inventing unavailable measurements."""

    def __init__(self, config: RoleConfig) -> None:
        self.config = config

    def fit(
        self,
        values: np.ndarray,
        labels: np.ndarray,
        feature_names: Sequence[str],
    ) -> RoleMapping:
        array = np.asarray(values, dtype=np.float64)
        target = np.asarray(labels, dtype=np.int64)
        roles: List[SemanticRole] = []
        for role_name in ("S", "C", "K", "E"):
            candidates = self._candidates(feature_names, role_name)
            if not candidates:
                continue
            indices = [index for index, _ in candidates]
            standardized, centers, scales = _robust_columns(array[:, indices])
            weights = self._sparse_weights(standardized, target)
            priorities = np.asarray([priority for _, priority in candidates], dtype=float)
            weights = weights * priorities
            if float(weights.sum()) <= 0.0:
                weights[int(np.argmax(priorities))] = 1.0
            order = np.argsort(-weights)[: self.config.max_components]
            selected = [index for index in order if weights[index] > 0.0]
            selected_weights = weights[selected]
            selected_weights = selected_weights / selected_weights.sum()
            components = []
            for local_index, weight in zip(selected, selected_weights):
                if (
                    len(selected_weights) > 1
                    and weight < self.config.minimum_component_weight
                ):
                    continue
                feature_index = indices[local_index]
                components.append(
                    RoleComponent(
                        feature_name=str(feature_names[feature_index]),
                        feature_index=int(feature_index),
                        weight=float(weight),
                        center=float(centers[local_index]),
                        scale=float(scales[local_index]),
                    )
                )
            if not components:
                local_index = int(order[0])
                feature_index = indices[local_index]
                components.append(
                    RoleComponent(
                        feature_name=str(feature_names[feature_index]),
                        feature_index=int(feature_index),
                        weight=1.0,
                        center=float(centers[local_index]),
                        scale=float(scales[local_index]),
                    )
                )
            total_weight = sum(component.weight for component in components)
            normalized_components = tuple(
                RoleComponent(
                    feature_name=component.feature_name,
                    feature_index=component.feature_index,
                    weight=component.weight / total_weight,
                    center=component.center,
                    scale=component.scale,
                )
                for component in components
            )
            roles.append(SemanticRole(role_name, normalized_components))
        if len(roles) < 2:
            raise ValueError(
                "At least two semantic roles are required for formula discovery"
            )
        return RoleMapping(tuple(roles))

    @staticmethod
    def _candidates(
        feature_names: Sequence[str], role_name: str
    ) -> List[Tuple[int, int]]:
        aliases = _ROLE_ALIASES[role_name]
        candidates = []
        for index, feature_name in enumerate(feature_names):
            priority = _candidate_priority(feature_name, aliases)
            if priority:
                candidates.append((index, priority))
        return candidates

    def _sparse_weights(self, values: np.ndarray, labels: np.ndarray) -> np.ndarray:
        if values.shape[1] == 1:
            return np.ones(1, dtype=np.float64)
        try:
            model = LogisticRegression(
                penalty="l1",
                solver="liblinear",
                C=self.config.l1_c,
                class_weight="balanced",
                max_iter=2000,
                random_state=0,
            )
            model.fit(values, labels)
            weights = np.abs(model.coef_[0]).astype(np.float64)
        except (ValueError, FloatingPointError):
            weights = np.zeros(values.shape[1], dtype=np.float64)
        if float(weights.sum()) <= 0.0:
            centered_labels = labels.astype(np.float64) - float(np.mean(labels))
            weights = np.abs(values.T @ centered_labels)
        return weights


def transform_roles_numpy(values: np.ndarray, mapping: RoleMapping) -> Dict[str, np.ndarray]:
    array = np.asarray(values, dtype=np.float64)
    transformed: Dict[str, np.ndarray] = {}
    for role in mapping.roles:
        result = np.zeros(len(array), dtype=np.float64)
        for component in role.components:
            column = np.log1p(np.maximum(array[:, component.feature_index], 0.0))
            result += component.weight * (
                (column - component.center) / component.scale
            )
        transformed[role.name] = result
    return transformed


def transform_roles_torch(values, mapping: RoleMapping):
    import torch

    transformed = {}
    for role in mapping.roles:
        result = torch.zeros(values.shape[0], dtype=values.dtype, device=values.device)
        for component in role.components:
            column = torch.log1p(torch.clamp(values[:, component.feature_index], min=0.0))
            result = result + component.weight * (
                (column - component.center) / component.scale
            )
        transformed[role.name] = result
    return transformed
