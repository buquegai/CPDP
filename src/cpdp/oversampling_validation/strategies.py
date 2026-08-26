from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import asdict
from typing import Any, Callable, Dict, Mapping, Protocol, Sequence

import numpy as np

from cpdp.oversampling.config import OversamplingConfig
from cpdp.oversampling.diffusion import ProjectConditionedTabularDDPM
from cpdp.oversampling.pipeline import IPGDOOversampler
from cpdp.oversampling.preprocessing import RobustAffineScaler
from cpdp.oversampling.types import AugmentationResult, freeze_array
from cpdp.types import LabeledProject, ProjectFeatures


class OversamplingStrategy(Protocol):
    def fit_resample(
        self, sources: Sequence[LabeledProject], seed: int
    ) -> AugmentationResult:
        ...


StrategyFactory = Callable[[Mapping[str, Any], Mapping[str, Any]], OversamplingStrategy]
_STRATEGIES: Dict[str, StrategyFactory] = {}


def deep_merge(
    base: Mapping[str, Any], overrides: Mapping[str, Any]
) -> Dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def register_strategy(name: str, factory: StrategyFactory) -> None:
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("Strategy name cannot be empty")
    _STRATEGIES[normalized] = factory


def available_strategy_kinds() -> tuple[str, ...]:
    return tuple(sorted(_STRATEGIES))


def _derived_seed(seed: int, project_key: str) -> int:
    payload = f"{seed}:{project_key}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=4).digest(), "big")


def _requested_count(project: LabeledProject, config: Mapping[str, Any]) -> int:
    clean = int(np.sum(project.labels == 0))
    defects = int(np.sum(project.labels == 1))
    target_ratio = float(config.get("target_minority_ratio", 1.0))
    requested = max(0, int(np.ceil(clean * target_ratio)) - defects)
    maximum = config.get("max_synthetic_per_project")
    return requested if maximum is None else min(requested, int(maximum))


def _append_synthetic(
    project: LabeledProject, synthetic_values: np.ndarray
) -> LabeledProject:
    values = np.asarray(synthetic_values, dtype=np.float64)
    if len(values) == 0:
        return project
    combined_values = np.vstack((project.features.values, values))
    labels = np.concatenate(
        (project.labels, np.ones(len(values), dtype=np.int64))
    )
    effort_index = project.features.feature_names.index(
        project.features.effort_name
    )
    effort = np.concatenate((project.features.effort, values[:, effort_index]))
    row_ids = np.concatenate(
        (
            project.features.row_ids,
            -np.arange(1, len(values) + 1, dtype=np.int64),
        )
    )
    cleaning = dict(project.features.cleaning)
    cleaning["synthetic_rows_added"] = int(len(values))
    features = ProjectFeatures(
        family=project.features.family,
        name=project.features.name,
        values=freeze_array(combined_values, np.float64),
        feature_names=project.features.feature_names,
        effort=freeze_array(effort, np.float64),
        effort_name=project.features.effort_name,
        row_ids=freeze_array(row_ids, np.int64),
        cleaning=cleaning,
    )
    return LabeledProject(features=features, labels=freeze_array(labels, np.int64))


class NoOversamplingStrategy:
    def fit_resample(
        self, sources: Sequence[LabeledProject], seed: int
    ) -> AugmentationResult:
        return AugmentationResult(
            projects=tuple(sources),
            metadata={
                "enabled": False,
                "name": "none",
                "seed": int(seed),
                "total_synthetic_selected": 0,
            },
        )


class RandomDefectOversamplingStrategy:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)

    def fit_resample(
        self, sources: Sequence[LabeledProject], seed: int
    ) -> AugmentationResult:
        projects = []
        diagnostics = []
        for project in sources:
            requested = _requested_count(project, self.config)
            defects = project.features.values[project.labels == 1]
            if requested and len(defects):
                random = np.random.default_rng(_derived_seed(seed, project.key))
                selected = defects[random.integers(0, len(defects), size=requested)]
            else:
                selected = np.empty((0, project.features.n_features), dtype=np.float64)
            projects.append(_append_synthetic(project, selected))
            diagnostics.append(
                {
                    "project": project.key,
                    "requested_synthetic": requested,
                    "selected_synthetic": int(len(selected)),
                }
            )
        return AugmentationResult(
            projects=tuple(projects),
            metadata={
                "enabled": True,
                "name": "random-defect-oversampling",
                "target_data_used": False,
                "configuration": self.config,
                "projects": diagnostics,
                "total_synthetic_selected": int(
                    sum(item["selected_synthetic"] for item in diagnostics)
                ),
            },
        )


class AdaptiveSMOTEStrategy:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        self.neighbors = int(self.config.get("smote_k_neighbors", 5))

    def fit_resample(
        self, sources: Sequence[LabeledProject], seed: int
    ) -> AugmentationResult:
        from imblearn.over_sampling import SMOTE

        projects = []
        diagnostics = []
        for project in sources:
            requested = _requested_count(project, self.config)
            defect_count = int(np.sum(project.labels == 1))
            selected = np.empty((0, project.features.n_features), dtype=np.float64)
            fallback = None
            if requested > 0 and defect_count > 1:
                desired = defect_count + requested
                neighbors = min(self.neighbors, defect_count - 1)
                sampler = SMOTE(
                    sampling_strategy={1: desired},
                    k_neighbors=neighbors,
                    random_state=_derived_seed(seed, project.key),
                )
                values, labels = sampler.fit_resample(
                    project.features.values, project.labels
                )
                selected = values[len(project.labels) :][labels[len(project.labels) :] == 1]
            elif requested > 0 and defect_count == 1:
                fallback = "single_defect_random_replication"
                defect = project.features.values[project.labels == 1][0]
                selected = np.repeat(defect[None, :], requested, axis=0)
            projects.append(_append_synthetic(project, selected))
            diagnostics.append(
                {
                    "project": project.key,
                    "requested_synthetic": requested,
                    "selected_synthetic": int(len(selected)),
                    "adaptive_neighbors": min(self.neighbors, max(0, defect_count - 1)),
                    "fallback": fallback,
                }
            )
        return AugmentationResult(
            projects=tuple(projects),
            metadata={
                "enabled": True,
                "name": "adaptive-smote",
                "target_data_used": False,
                "configuration": self.config,
                "projects": diagnostics,
                "total_synthetic_selected": int(
                    sum(item["selected_synthetic"] for item in diagnostics)
                ),
            },
        )


class ProjectConditionedTabDDPMStrategy:
    """Pure TabDDPM baseline without IPGDO's KNN candidate filter."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = OversamplingConfig.from_mapping(config)

    def fit_resample(
        self, sources: Sequence[LabeledProject], seed: int
    ) -> AugmentationResult:
        projects = tuple(sources)
        self._validate_sources(projects)
        requested = tuple(_requested_count(project, asdict(self.config)) for project in projects)
        if sum(requested) == 0:
            return AugmentationResult(
                projects=projects,
                metadata={
                    "enabled": False,
                    "name": "TabDDPM",
                    "reason": "all_source_projects_already_meet_target_ratio",
                    "target_data_used": False,
                    "configuration": asdict(self.config),
                    "total_synthetic_selected": 0,
                },
            )

        defect_batches = tuple(
            project.features.values[project.labels == 1] for project in projects
        )
        scalers = tuple(RobustAffineScaler.fit(values) for values in defect_batches)
        normalized = np.vstack(
            [
                np.clip(
                    scaler.transform(values),
                    -self.config.diffusion.normalized_clip,
                    self.config.diffusion.normalized_clip,
                )
                for values, scaler in zip(defect_batches, scalers)
            ]
        ).astype(np.float32)
        project_ids = np.concatenate(
            [
                np.full(len(values), index, dtype=np.int64)
                for index, values in enumerate(defect_batches)
            ]
        )
        diffusion = ProjectConditionedTabularDDPM(
            input_dim=normalized.shape[1],
            project_count=len(projects),
            config=self.config.diffusion,
            device=self.config.device,
        )
        diffusion_metadata = diffusion.fit(normalized, project_ids, seed)

        augmented = []
        diagnostics = []
        for project_index, (project, needed, scaler) in enumerate(
            zip(projects, requested, scalers)
        ):
            if needed:
                generated = scaler.inverse_transform(
                    diffusion.sample_defects(needed, project_index)
                )
                if not np.isfinite(generated).all():
                    raise RuntimeError(
                        f"TabDDPM generated non-finite values for {project.key}"
                    )
            else:
                generated = np.empty(
                    (0, project.features.n_features), dtype=np.float64
                )
            augmented.append(_append_synthetic(project, generated))
            diagnostics.append(
                {
                    "project": project.key,
                    "requested_synthetic": int(needed),
                    "generated_candidates": int(len(generated)),
                    "selected_synthetic": int(len(generated)),
                    "knn": {"enabled": False, "reason": "tabddpm_baseline"},
                }
            )
            print(
                "PROGRESS tabddpm "
                f"project={project.key} requested={needed} selected={len(generated)}",
                flush=True,
            )

        return AugmentationResult(
            projects=tuple(augmented),
            metadata={
                "enabled": True,
                "name": "TabDDPM",
                "display_name": "Project-Conditioned Tabular DDPM",
                "target_data_used": False,
                "training_data": "real_source_defects_only",
                "architecture": ["project-conditioned tabular DDPM"],
                "configuration": asdict(self.config),
                "diffusion": diffusion_metadata,
                "projects": diagnostics,
                "total_synthetic_selected": int(sum(requested)),
            },
        )

    @staticmethod
    def _validate_sources(projects: Sequence[LabeledProject]) -> None:
        if not projects:
            raise ValueError("TabDDPM requires at least one source project")
        family = projects[0].features.family
        schema = projects[0].features.feature_names
        keys = set()
        for project in projects:
            if project.key in keys:
                raise ValueError(f"Duplicate source project: {project.key}")
            keys.add(project.key)
            if project.features.family != family:
                raise ValueError("All source projects must belong to one dataset family")
            if project.features.feature_names != schema:
                raise ValueError("All source projects must use the same feature schema")
            if set(np.unique(project.labels).tolist()) != {0, 1}:
                raise ValueError(f"Source project must contain both classes: {project.key}")


class CurrentPipelineStrategy:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = deepcopy(dict(config))

    def fit_resample(
        self, sources: Sequence[LabeledProject], seed: int
    ) -> AugmentationResult:
        oversampler = IPGDOOversampler.from_mapping(self.config)
        return oversampler.fit_resample(sources, seed)


def _none_factory(
    base: Mapping[str, Any], overrides: Mapping[str, Any]
) -> OversamplingStrategy:
    return NoOversamplingStrategy()


def _random_factory(
    base: Mapping[str, Any], overrides: Mapping[str, Any]
) -> OversamplingStrategy:
    return RandomDefectOversamplingStrategy(deep_merge(base, overrides))


def _smote_factory(
    base: Mapping[str, Any], overrides: Mapping[str, Any]
) -> OversamplingStrategy:
    return AdaptiveSMOTEStrategy(deep_merge(base, overrides))


def _current_factory(
    base: Mapping[str, Any], overrides: Mapping[str, Any]
) -> OversamplingStrategy:
    return CurrentPipelineStrategy(deep_merge(base, overrides))


def _tabddpm_factory(
    base: Mapping[str, Any], overrides: Mapping[str, Any]
) -> OversamplingStrategy:
    return ProjectConditionedTabDDPMStrategy(deep_merge(base, overrides))


register_strategy("none", _none_factory)
register_strategy("random", _random_factory)
register_strategy("smote", _smote_factory)
register_strategy("current", _current_factory)
register_strategy("full", _current_factory)
register_strategy("tabddpm", _tabddpm_factory)


def create_strategy(
    kind: str,
    base_config: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> OversamplingStrategy:
    normalized = kind.strip().lower()
    try:
        factory = _STRATEGIES[normalized]
    except KeyError as exc:
        raise KeyError(
            f"Unknown strategy kind {kind!r}; available: {', '.join(available_strategy_kinds())}"
        ) from exc
    return factory(base_config, overrides)


def effective_strategy_config(
    kind: str,
    base_config: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> Mapping[str, Any]:
    normalized = kind.strip().lower()
    if normalized == "none":
        return {"kind": normalized}
    return {"kind": normalized, "config": deep_merge(base_config, overrides)}
