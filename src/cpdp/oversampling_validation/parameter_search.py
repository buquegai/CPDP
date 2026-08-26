from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

from cpdp.config import load_config
from cpdp.metrics import METRIC_NAMES
from cpdp.oversampling_validation.config import load_validation_config
from cpdp.oversampling_validation.parameter_plot import (
    create_parameter_comparison_figure,
)
from cpdp.oversampling_validation.runner import OversamplingValidationRunner
from cpdp.oversampling_validation.store import ValidationStore


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "ipgdo_parameter_search.yaml"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _expand_path(value: str, base: Path) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(value))
    path = Path(expanded)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _float_tuple(values: Iterable[Any], name: str) -> Tuple[float, ...]:
    parsed = tuple(float(value) for value in values)
    if len(parsed) < 2:
        raise ValueError(f"{name} must contain at least two values")
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"{name} must not contain duplicate values")
    return parsed


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(temporary, path)


def _atomic_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(
            dict(payload),
            handle,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def _number_text(value: float) -> str:
    return format(float(value), ".12g")


def _number_slug(value: float) -> str:
    return _number_text(value).replace("-", "m").replace(".", "p")


def _source_signature(project_root: Path) -> str:
    digest = hashlib.sha256()
    source_roots = (
        project_root / "src" / "cpdp" / "oversampling",
        project_root / "src" / "cpdp" / "oversampling_validation",
    )
    for source_root in source_roots:
        for path in sorted(source_root.rglob("*.py")):
            digest.update(
                str(path.relative_to(project_root)).replace("\\", "/").encode("utf-8")
            )
            with path.open("rb") as handle:
                while True:
                    block = handle.read(1024 * 1024)
                    if not block:
                        break
                    digest.update(block)
    return digest.hexdigest()[:12]


@dataclass(frozen=True)
class ParameterSet:
    target_minority_ratio: float
    physics_timestep_fraction: float
    condition_dropout: float

    def validate(self) -> None:
        if self.target_minority_ratio <= 0:
            raise ValueError("target_minority_ratio must be positive")
        if not 0 < self.physics_timestep_fraction <= 1:
            raise ValueError("physics_timestep_fraction must be in (0, 1]")
        if not 0 <= self.condition_dropout < 1:
            raise ValueError("condition_dropout must be in [0, 1)")

    @property
    def trial_id(self) -> str:
        return (
            f"ratio_{_number_slug(self.target_minority_ratio)}"
            f"__physics_steps_{_number_slug(self.physics_timestep_fraction)}"
            f"__condition_dropout_{_number_slug(self.condition_dropout)}"
        )

    @property
    def estimated_cost(self) -> float:
        return float(self.target_minority_ratio)

    def to_dict(self) -> Dict[str, float]:
        return {
            "target_minority_ratio": float(self.target_minority_ratio),
            "physics_timestep_fraction": float(self.physics_timestep_fraction),
            "condition_dropout": float(self.condition_dropout),
        }


@dataclass(frozen=True)
class ParameterGrids:
    target_minority_ratio: Tuple[float, ...]
    physics_timestep_fraction: Tuple[float, ...]
    condition_dropout: Tuple[float, ...]


@dataclass(frozen=True)
class SelectionConfig:
    metrics: Tuple[str, ...]
    top_values_per_stage: int
    safety_tolerances: Mapping[str, float]


@dataclass(frozen=True)
class ExecutionConfig:
    workers: int
    fail_fast: bool
    elastic_gpu: Mapping[str, Any]

    def validate(self) -> None:
        if self.workers < 1:
            raise ValueError("execution.workers must be positive")


@dataclass(frozen=True)
class ParameterSearchConfig:
    source_path: Path
    project_root: Path
    experiment_config: Path
    output_dir: Path
    validation_output_dir: Path
    runner_script: Path
    oversampling_method: str
    oversampling_profile: str
    profile_name: str
    variant_name: str
    variant_kind: str
    initial: ParameterSet
    grids: ParameterGrids
    screening_seeds: Tuple[int, ...]
    confirmation_seeds: Tuple[int, ...]
    fixed_overrides: Mapping[str, Any]
    source_cv_folds: int
    source_cv_validation_fraction: float
    classifier: Mapping[str, Any]
    quality: Mapping[str, Any]
    selection: SelectionConfig
    execution: ExecutionConfig

    @property
    def all_seeds(self) -> Tuple[int, ...]:
        return self.screening_seeds + self.confirmation_seeds

    def validate(self) -> None:
        self.initial.validate()
        for value in self.grids.target_minority_ratio:
            ParameterSet(
                value,
                self.initial.physics_timestep_fraction,
                self.initial.condition_dropout,
            ).validate()
        for value in self.grids.physics_timestep_fraction:
            ParameterSet(
                self.initial.target_minority_ratio,
                value,
                self.initial.condition_dropout,
            ).validate()
        for value in self.grids.condition_dropout:
            ParameterSet(
                self.initial.target_minority_ratio,
                self.initial.physics_timestep_fraction,
                value,
            ).validate()
        if self.initial.target_minority_ratio not in self.grids.target_minority_ratio:
            raise ValueError("initial target_minority_ratio must occur in its grid")
        if (
            self.initial.physics_timestep_fraction
            not in self.grids.physics_timestep_fraction
        ):
            raise ValueError(
                "initial physics_timestep_fraction must occur in its grid"
            )
        if self.initial.condition_dropout not in self.grids.condition_dropout:
            raise ValueError("initial condition_dropout must occur in its grid")
        if self.source_cv_folds < 2:
            raise ValueError("validation.source_cv.folds must be at least 2")
        if not 0 < self.source_cv_validation_fraction < 0.5:
            raise ValueError(
                "validation.source_cv.validation_fraction must be in (0, 0.5)"
            )
        if not self.screening_seeds or not self.confirmation_seeds:
            raise ValueError("screening and confirmation seed lists cannot be empty")
        if set(self.screening_seeds) & set(self.confirmation_seeds):
            raise ValueError("screening and confirmation seeds must be disjoint")
        if len(set(self.all_seeds)) != len(self.all_seeds):
            raise ValueError("all search seeds must be unique")
        unknown_metrics = sorted(set(self.selection.metrics) - set(METRIC_NAMES))
        if unknown_metrics:
            raise ValueError(f"Unknown selection metrics: {', '.join(unknown_metrics)}")
        if self.selection.top_values_per_stage != 2:
            raise ValueError("top_values_per_stage must be 2 for the 2x2x2 check")
        unknown_safety = sorted(
            set(self.selection.safety_tolerances) - set(self.selection.metrics)
        )
        if unknown_safety:
            raise ValueError(f"Unknown safety metrics: {', '.join(unknown_safety)}")
        if any(value < 0 for value in self.selection.safety_tolerances.values()):
            raise ValueError("Safety tolerances cannot be negative")
        self.execution.validate()

    def signature(self) -> str:
        payload = {
            "experiment_config": str(self.experiment_config),
            "oversampling_method": self.oversampling_method,
            "oversampling_profile": self.oversampling_profile,
            "initial": self.initial.to_dict(),
            "grids": asdict(self.grids),
            "screening_seeds": self.screening_seeds,
            "confirmation_seeds": self.confirmation_seeds,
            "fixed_overrides": self.fixed_overrides,
            "source_cv_folds": self.source_cv_folds,
            "source_cv_validation_fraction": self.source_cv_validation_fraction,
            "classifier": self.classifier,
            "quality": self.quality,
            "selection": asdict(self.selection),
        }
        encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]


def load_parameter_search_config(path: Path) -> ParameterSearchConfig:
    source_path = path.resolve()
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping) or raw.get("schema_version") != 1:
        raise ValueError("Unsupported IPGDO parameter-search schema_version")

    project_root = source_path.parent.parent
    paths = _mapping(raw.get("paths"), "paths")
    experiment_config = _expand_path(
        str(paths.get("experiment_config", "experiment.yaml")), source_path.parent
    )
    output_dir = _expand_path(
        str(paths.get("output_dir", "outputs/ipgdo_parameter_search")),
        project_root,
    )
    validation_output_dir = _expand_path(
        str(
            paths.get(
                "validation_output_dir",
                "outputs/ipgdo_parameter_search/validation_runs",
            )
        ),
        project_root,
    )
    runner_script = _expand_path(
        str(paths.get("runner_script", "run_oversampling_validation.py")),
        project_root,
    )

    method = _mapping(raw.get("method"), "method")
    initial_values = _mapping(raw.get("initial_values"), "initial_values")
    initial = ParameterSet(
        target_minority_ratio=float(initial_values["target_minority_ratio"]),
        physics_timestep_fraction=float(
            initial_values["physics_timestep_fraction"]
        ),
        condition_dropout=float(initial_values["condition_dropout"]),
    )
    grids_raw = _mapping(raw.get("parameter_grids"), "parameter_grids")
    grids = ParameterGrids(
        target_minority_ratio=_float_tuple(
            grids_raw.get("target_minority_ratio", ()),
            "parameter_grids.target_minority_ratio",
        ),
        physics_timestep_fraction=_float_tuple(
            grids_raw.get("physics_timestep_fraction", ()),
            "parameter_grids.physics_timestep_fraction",
        ),
        condition_dropout=_float_tuple(
            grids_raw.get("condition_dropout", ()),
            "parameter_grids.condition_dropout",
        ),
    )
    seeds = _mapping(raw.get("seeds"), "seeds")
    screening_seeds = tuple(int(seed) for seed in seeds.get("screening", ()))
    confirmation_seeds = tuple(int(seed) for seed in seeds.get("confirmation", ()))

    validation = _mapping(raw.get("validation"), "validation")
    source_cv = _mapping(validation.get("source_cv"), "validation.source_cv")
    selection_raw = _mapping(raw.get("selection"), "selection")
    safety = {
        str(name): float(value)
        for name, value in _mapping(
            selection_raw.get("maximum_allowed_decline"),
            "selection.maximum_allowed_decline",
        ).items()
    }
    selection = SelectionConfig(
        metrics=tuple(str(name) for name in selection_raw.get("metrics", ())),
        top_values_per_stage=int(
            selection_raw.get("top_values_per_stage", 2)
        ),
        safety_tolerances=safety,
    )
    execution_raw = _mapping(raw.get("execution"), "execution")
    execution = ExecutionConfig(
        workers=int(execution_raw.get("workers", 4)),
        fail_fast=bool(execution_raw.get("fail_fast", True)),
        elastic_gpu=dict(
            _mapping(execution_raw.get("elastic_gpu"), "execution.elastic_gpu")
        ),
    )
    config = ParameterSearchConfig(
        source_path=source_path,
        project_root=project_root,
        experiment_config=experiment_config,
        output_dir=output_dir,
        validation_output_dir=validation_output_dir,
        runner_script=runner_script,
        oversampling_method=str(method.get("oversampling_method", "ipgdo_kan_jepa")),
        oversampling_profile=str(method.get("oversampling_profile", "full")),
        profile_name=str(method.get("generated_profile_name", "search")),
        variant_name=str(method.get("variant_name", "full")),
        variant_kind=str(method.get("variant_kind", "current")),
        initial=initial,
        grids=grids,
        screening_seeds=screening_seeds,
        confirmation_seeds=confirmation_seeds,
        fixed_overrides=dict(
            _mapping(raw.get("fixed_overrides"), "fixed_overrides")
        ),
        source_cv_folds=int(source_cv.get("folds", 3)),
        source_cv_validation_fraction=float(
            source_cv.get("validation_fraction", 0.2)
        ),
        classifier=dict(_mapping(validation.get("classifier"), "validation.classifier")),
        quality=dict(_mapping(validation.get("quality"), "validation.quality")),
        selection=selection,
        execution=execution,
    )
    config.validate()
    if not config.experiment_config.is_file():
        raise FileNotFoundError(
            f"Experiment configuration not found: {config.experiment_config}"
        )
    if not config.runner_script.is_file():
        raise FileNotFoundError(f"Validation runner not found: {config.runner_script}")
    return config


def stage_one_candidates(config: ParameterSearchConfig) -> Tuple[ParameterSet, ...]:
    return tuple(
        ParameterSet(
            target_minority_ratio=value,
            physics_timestep_fraction=config.initial.physics_timestep_fraction,
            condition_dropout=config.initial.condition_dropout,
        )
        for value in config.grids.target_minority_ratio
    )


def stage_two_candidates(
    config: ParameterSearchConfig, best_target_ratio: float
) -> Tuple[ParameterSet, ...]:
    return tuple(
        ParameterSet(
            target_minority_ratio=best_target_ratio,
            physics_timestep_fraction=value,
            condition_dropout=config.initial.condition_dropout,
        )
        for value in config.grids.physics_timestep_fraction
    )


def stage_three_candidates(
    config: ParameterSearchConfig,
    best_target_ratio: float,
    best_physics_timestep_fraction: float,
) -> Tuple[ParameterSet, ...]:
    return tuple(
        ParameterSet(
            target_minority_ratio=best_target_ratio,
            physics_timestep_fraction=best_physics_timestep_fraction,
            condition_dropout=value,
        )
        for value in config.grids.condition_dropout
    )


def interaction_candidates(
    target_ratios: Sequence[float],
    physics_timestep_fractions: Sequence[float],
    condition_dropouts: Sequence[float],
) -> Tuple[ParameterSet, ...]:
    return tuple(
        ParameterSet(target_ratio, physics_timestep_fraction, condition_dropout)
        for target_ratio in target_ratios
        for physics_timestep_fraction in physics_timestep_fractions
        for condition_dropout in condition_dropouts
    )


def unique_parameter_sets(
    parameter_sets: Iterable[ParameterSet],
) -> Tuple[ParameterSet, ...]:
    return tuple({item.trial_id: item for item in parameter_sets}.values())


def aggregate_source_cv_project_metrics(
    frame: pd.DataFrame,
    *,
    trial_id: str,
    variant: str,
    seeds: Sequence[int],
    folds: Sequence[int],
    projects: Sequence[Tuple[str, str]],
    metrics: Sequence[str],
) -> pd.DataFrame:
    metric_columns = [f"augmented_{metric}" for metric in metrics]
    required = {
        "protocol",
        "variant",
        "family",
        "evaluated_project",
        "seed",
        "fold",
        *metric_columns,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"Trial {trial_id} summary is missing: {', '.join(missing)}"
        )
    selected = frame[
        (frame["protocol"] == "source_cv")
        & (frame["variant"] == variant)
        & (frame["seed"].isin([int(seed) for seed in seeds]))
        & (frame["fold"].isin([int(fold) for fold in folds]))
    ].copy()
    selected["target"] = selected["evaluated_project"].astype(str).map(
        lambda value: value.split("/", 1)[-1]
    )
    per_fold = (
        selected.groupby(
            ["family", "target", "seed", "fold"], as_index=False
        )[metric_columns]
        .mean(numeric_only=True)
    )
    expected_projects = {(str(family), str(project)) for family, project in projects}
    actual_projects = set(
        zip(per_fold["family"].astype(str), per_fold["target"].astype(str))
    )
    expected_rows = len(expected_projects) * len(folds) * len(seeds)
    if actual_projects != expected_projects or len(per_fold) != expected_rows:
        raise RuntimeError(
            f"Trial {trial_id} has {len(per_fold)}/{expected_rows} "
            "source-project-fold-seed result rows"
        )
    project_means = (
        per_fold.groupby(["family", "target"], as_index=False)[metric_columns]
        .mean(numeric_only=True)
        .rename(
            columns={f"augmented_{metric}": metric for metric in metrics}
        )
    )
    project_means.insert(0, "trial_id", trial_id)
    return project_means


def rank_candidates(
    candidate_metrics: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
    parameters: Mapping[str, ParameterSet],
    metrics: Sequence[str],
    safety_tolerances: Mapping[str, float],
) -> pd.DataFrame:
    required = {"trial_id", "family", "target", *metrics}
    missing = sorted(required - set(candidate_metrics.columns))
    if missing:
        raise ValueError(f"Candidate metrics are missing columns: {', '.join(missing)}")
    baseline_missing = sorted({"family", "target", *metrics} - set(baseline_metrics.columns))
    if baseline_missing:
        raise ValueError(
            f"Baseline metrics are missing columns: {', '.join(baseline_missing)}"
        )
    candidate_ids = tuple(dict.fromkeys(candidate_metrics["trial_id"].astype(str)))
    if set(candidate_ids) != set(parameters):
        raise ValueError("Candidate metric trial IDs do not match parameter definitions")

    expected_targets: Optional[set[Tuple[str, str]]] = None
    for trial_id, frame in candidate_metrics.groupby("trial_id"):
        targets = set(zip(frame["family"].astype(str), frame["target"].astype(str)))
        if expected_targets is None:
            expected_targets = targets
        elif targets != expected_targets:
            raise ValueError(f"Target projects are incomplete for trial {trial_id}")
    baseline_targets = set(
        zip(baseline_metrics["family"].astype(str), baseline_metrics["target"].astype(str))
    )
    if expected_targets != baseline_targets:
        raise ValueError("Baseline and candidate project sets do not match")

    rank_values: Dict[str, list[float]] = {trial_id: [] for trial_id in candidate_ids}
    for _, target_frame in candidate_metrics.groupby(["family", "target"], sort=False):
        indexed = target_frame.set_index("trial_id")
        for metric in metrics:
            values = pd.to_numeric(indexed[metric], errors="coerce")
            finite = values[np.isfinite(values)]
            if finite.empty:
                continue
            ranked = finite.rank(method="average", ascending=False)
            worst_rank = float(len(finite) + 1)
            for trial_id in candidate_ids:
                value = ranked.get(trial_id, worst_rank)
                rank_values[trial_id].append(float(value))

    baseline_macro = {
        metric: float(pd.to_numeric(baseline_metrics[metric], errors="coerce").mean())
        for metric in metrics
    }
    rows = []
    for trial_id in candidate_ids:
        frame = candidate_metrics[candidate_metrics["trial_id"] == trial_id]
        item = parameters[trial_id]
        row: Dict[str, Any] = {
            "trial_id": trial_id,
            **item.to_dict(),
            "mean_rank": float(np.mean(rank_values[trial_id])),
            "ranked_cells": int(len(rank_values[trial_id])),
            "estimated_cost": item.estimated_cost,
        }
        for metric in metrics:
            row[f"macro_{metric}"] = float(
                pd.to_numeric(frame[metric], errors="coerce").mean()
            )
        violations = []
        for metric, tolerance in safety_tolerances.items():
            decline = baseline_macro[metric] - row[f"macro_{metric}"]
            row[f"decline_{metric}"] = float(decline)
            if not np.isfinite(decline) or decline > float(tolerance) + 1e-12:
                violations.append(metric)
        row["safety_eligible"] = not violations
        row["safety_violations"] = ",".join(violations)
        rows.append(row)

    ranking = pd.DataFrame(rows)
    if ranking.empty or not ranking["safety_eligible"].any():
        raise RuntimeError("No candidate satisfies the configured safety constraints")
    for metric in ("mcc", "gmean"):
        column = f"macro_{metric}"
        if column not in ranking:
            ranking[column] = np.nan
    ranking = ranking.sort_values(
        [
            "safety_eligible",
            "mean_rank",
            "macro_mcc",
            "macro_gmean",
            "estimated_cost",
            "trial_id",
        ],
        ascending=[False, True, False, False, True, True],
        na_position="last",
    ).reset_index(drop=True)
    ranking.insert(0, "selection_position", np.arange(1, len(ranking) + 1))
    return ranking


@dataclass(frozen=True)
class TrialContext:
    parameters: ParameterSet
    config_path: Path
    fingerprint: str
    store: ValidationStore
    projects: Tuple[Tuple[str, str], ...]
    families: Tuple[str, ...]
    folds: Tuple[int, ...]


class ParameterSearchEngine:
    def __init__(
        self,
        config: ParameterSearchConfig,
        *,
        workers: Optional[int] = None,
        python_executable: Optional[Path] = None,
    ) -> None:
        self.config = config
        self.workers = int(workers or config.execution.workers)
        if self.workers < 1:
            raise ValueError("workers must be positive")
        self.python_executable = Path(python_executable or sys.executable).resolve()
        self.generated_config_dir = config.output_dir / "generated_configs"
        self.source_signature = _source_signature(config.project_root)
        self.state_path = config.output_dir / "search_state.json"
        self.failures_path = config.output_dir / "failures.csv"
        self.state = self._initial_state()
        self._trial_contexts: Dict[str, TrialContext] = {}

    def _initial_state(self) -> Dict[str, Any]:
        state = {
            "schema_version": 1,
            "search_signature": self.config.signature(),
            "config_path": str(self.config.source_path),
            "status": "not_started",
            "started_at": None,
            "updated_at": _utc_now(),
            "stages": {},
            "trials": {},
        }
        if self.state_path.is_file():
            try:
                with self.state_path.open("r", encoding="utf-8") as handle:
                    existing = json.load(handle)
                if existing.get("search_signature") == self.config.signature():
                    state = existing
            except (OSError, json.JSONDecodeError):
                pass
        return state

    def _save_state(self) -> None:
        self.state["updated_at"] = _utc_now()
        _atomic_json(self.state_path, self.state)

    def _effective_overrides(self, parameters: ParameterSet) -> Dict[str, Any]:
        overrides = deepcopy(dict(self.config.fixed_overrides))
        overrides["target_minority_ratio"] = parameters.target_minority_ratio
        diffusion = dict(_mapping(overrides.get("diffusion"), "fixed_overrides.diffusion"))
        diffusion["physics_timestep_fraction"] = (
            parameters.physics_timestep_fraction
        )
        diffusion["condition_dropout"] = parameters.condition_dropout
        overrides["diffusion"] = diffusion
        return overrides

    def _trial_mapping(self, parameters: ParameterSet) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "experiment_config": str(self.config.experiment_config),
            "output_dir": str(self.config.validation_output_dir),
            "oversampling_method": self.config.oversampling_method,
            "variants": {
                self.config.variant_name: {
                    "kind": self.config.variant_kind,
                    "overrides": self._effective_overrides(parameters),
                }
            },
            "profiles": {
                self.config.profile_name: {
                    "oversampling_profile": self.config.oversampling_profile,
                    "seeds": list(self.config.all_seeds),
                    "variants": [self.config.variant_name],
                    "source_cv": {
                        "enabled": True,
                        "folds": self.config.source_cv_folds,
                        "validation_fraction": (
                            self.config.source_cv_validation_fraction
                        ),
                    },
                    "cpdp": {"enabled": False, "validation_fraction": 0.2},
                    "classifier": dict(self.config.classifier),
                    "quality": dict(self.config.quality),
                }
            },
        }

    def _context(self, parameters: ParameterSet) -> TrialContext:
        cached = self._trial_contexts.get(parameters.trial_id)
        if cached is not None:
            return cached
        config_path = self.generated_config_dir / (
            f"{parameters.trial_id}__code_{self.source_signature}.yaml"
        )
        _atomic_yaml(config_path, self._trial_mapping(parameters))
        validation_config = load_validation_config(config_path)
        runner = OversamplingValidationRunner(validation_config)
        fingerprint = runner.fingerprint(self.config.profile_name)
        context = TrialContext(
            parameters=parameters,
            config_path=config_path,
            fingerprint=fingerprint,
            store=ValidationStore(
                validation_config.output_dir,
                fingerprint,
                self.config.profile_name,
            ),
            projects=tuple(validation_config.experiment.targets()),
            families=tuple(
                family.name for family in validation_config.experiment.data.families
            ),
            folds=tuple(
                range(
                    validation_config.profile(
                        self.config.profile_name
                    ).source_cv.folds
                )
            ),
        )
        self._trial_contexts[parameters.trial_id] = context
        return context

    def _completed_seeds(
        self, context: TrialContext, seeds: Sequence[int]
    ) -> Tuple[int, ...]:
        completed = []
        for seed in seeds:
            if all(
                context.store.is_complete(
                    context.store.run_path(
                        "source_cv",
                        self.config.variant_name,
                        family,
                        int(seed),
                        fold=fold,
                    )
                )
                for family in context.families
                for fold in context.folds
            ):
                completed.append(int(seed))
        return tuple(completed)

    def _failure_records(
        self, context: TrialContext, seeds: Sequence[int]
    ) -> list[Dict[str, Any]]:
        records = []
        for seed in seeds:
            for family in context.families:
                for fold in context.folds:
                    path = context.store.run_path(
                        "source_cv",
                        self.config.variant_name,
                        family,
                        int(seed),
                        fold=fold,
                    )
                    if not path.is_file():
                        continue
                    try:
                        with path.open("r", encoding="utf-8") as handle:
                            payload = json.load(handle)
                    except (OSError, json.JSONDecodeError):
                        continue
                    if payload.get("status") != "failure":
                        continue
                    records.append(
                        {
                            "trial_id": context.parameters.trial_id,
                            **context.parameters.to_dict(),
                            "family": family,
                            "fold": int(fold),
                            "seed": int(seed),
                            "error_type": payload.get("error_type"),
                            "error_message": payload.get("error_message"),
                            "result_path": str(path),
                        }
                    )
        return records

    def _subprocess_environment(self) -> Dict[str, str]:
        environment = os.environ.copy()
        environment["PYTHONUTF8"] = "1"
        environment["PYTHONUNBUFFERED"] = "1"
        elastic = self.config.execution.elastic_gpu
        settings = {
            "CPDP_OVERSAMPLING_VALIDATION_WORKERS": str(self.workers),
            "CPDP_OVERSAMPLING_ELASTIC_GPU": "1"
            if bool(elastic.get("enabled", True))
            else "0",
            "CPDP_GPU_LAUNCH_UTILIZATION_LIMIT": str(
                elastic.get("launch_utilization_limit", 75)
            ),
            "CPDP_GPU_MIN_FREE_MEMORY_MB": str(
                elastic.get("minimum_free_memory_mb", 2048)
            ),
            "CPDP_GPU_MAX_TEMPERATURE_C": str(
                elastic.get("maximum_temperature_c", 80)
            ),
            "CPDP_GPU_POLL_SECONDS": str(elastic.get("poll_seconds", 2)),
            "CPDP_GPU_RAMP_UP_SECONDS": str(elastic.get("ramp_up_seconds", 8)),
            "CPDP_GPU_PAUSE_PROCESSES": ";".join(
                str(name) for name in elastic.get("pause_process_names", ())
            ),
        }
        environment.update(settings)
        return environment

    def _record_trial(
        self,
        context: TrialContext,
        requested_seeds: Sequence[int],
        status: str,
        stage: str,
    ) -> None:
        completed = self._completed_seeds(context, self.config.all_seeds)
        self.state.setdefault("trials", {})[context.parameters.trial_id] = {
            "parameters": context.parameters.to_dict(),
            "stage": stage,
            "status": status,
            "config_path": str(context.config_path),
            "fingerprint": context.fingerprint,
            "output_root": str(context.store.root),
            "requested_seeds": [int(seed) for seed in requested_seeds],
            "completed_seeds": list(completed),
        }
        self._save_state()

    def execute_trial(
        self,
        parameters: ParameterSet,
        seeds: Sequence[int],
        stage: str,
    ) -> TrialContext:
        context = self._context(parameters)
        completed = self._completed_seeds(context, seeds)
        if len(completed) == len(seeds):
            context.store.summarize()
            self._record_trial(context, seeds, "reused", stage)
            print(
                f"SEARCH_REUSE stage={stage} trial={parameters.trial_id} "
                f"seeds={len(seeds)}",
                flush=True,
            )
            return context

        print(
            f"SEARCH_RUN stage={stage} trial={parameters.trial_id} "
            f"completed_seeds={len(completed)}/{len(seeds)} workers={self.workers}",
            flush=True,
        )
        command = [
            str(self.python_executable),
            str(self.config.runner_script),
            "--config",
            str(context.config_path),
            "--workers",
            str(self.workers),
            "run",
            "--profile",
            self.config.profile_name,
            "--protocol",
            "source_cv",
            "--variant",
            self.config.variant_name,
        ]
        for seed in seeds:
            command.extend(("--seed", str(int(seed))))
        if self.config.execution.fail_fast:
            command.append("--fail-fast")
        completed_process = subprocess.run(
            command,
            cwd=self.config.project_root,
            env=self._subprocess_environment(),
            check=False,
        )
        completed = self._completed_seeds(context, seeds)
        if completed_process.returncode != 0 or len(completed) != len(seeds):
            failures = self._failure_records(context, seeds)
            if failures:
                existing = (
                    pd.read_csv(self.failures_path)
                    if self.failures_path.is_file()
                    else pd.DataFrame()
                )
                combined = pd.concat((existing, pd.DataFrame(failures)), ignore_index=True)
                combined = combined.drop_duplicates(
                    ["trial_id", "family", "fold", "seed"], keep="last"
                )
                _atomic_csv(self.failures_path, combined)
            self._record_trial(context, seeds, "failed", stage)
            missing_count = len(context.families) * len(context.folds) * len(seeds) - sum(
                context.store.is_complete(
                    context.store.run_path(
                        "source_cv",
                        self.config.variant_name,
                        family,
                        int(seed),
                        fold=fold,
                    )
                )
                for seed in seeds
                for family in context.families
                for fold in context.folds
            )
            raise RuntimeError(
                f"Trial {parameters.trial_id} is incomplete: "
                f"return_code={completed_process.returncode}, missing_tasks={missing_count}; "
                f"see {self.failures_path}"
            )
        context.store.summarize()
        self._record_trial(context, seeds, "completed", stage)
        return context

    def _source_cv_project_metrics(
        self, parameters: ParameterSet, seeds: Sequence[int]
    ) -> pd.DataFrame:
        context = self._context(parameters)
        summary_path = context.store.root / "summary" / "project_utility_runs.csv"
        context.store.summarize()
        frame = pd.read_csv(summary_path)
        return aggregate_source_cv_project_metrics(
            frame,
            trial_id=parameters.trial_id,
            variant=self.config.variant_name,
            seeds=seeds,
            folds=context.folds,
            projects=context.projects,
            metrics=self.config.selection.metrics,
        )

    def _run_stage(
        self,
        stage_name: str,
        candidates: Sequence[ParameterSet],
        baseline_metrics: Optional[pd.DataFrame],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        unique_candidates = unique_parameter_sets(candidates)
        for parameters in unique_candidates:
            self.execute_trial(parameters, self.config.screening_seeds, stage_name)
        frames = [
            self._source_cv_project_metrics(parameters, self.config.screening_seeds)
            for parameters in unique_candidates
        ]
        candidate_metrics = pd.concat(frames, ignore_index=True)
        baseline = baseline_metrics
        if baseline is None:
            baseline = self._source_cv_project_metrics(
                self.config.initial, self.config.screening_seeds
            )
        parameters_by_id = {
            parameters.trial_id: parameters for parameters in unique_candidates
        }
        ranking = rank_candidates(
            candidate_metrics,
            baseline,
            parameters_by_id,
            self.config.selection.metrics,
            self.config.selection.safety_tolerances,
        )
        ranking_path = self.config.output_dir / f"{stage_name}_ranking.csv"
        _atomic_csv(ranking_path, ranking)
        eligible = ranking[ranking["safety_eligible"]]
        top = eligible.head(self.config.selection.top_values_per_stage)
        if len(top) < self.config.selection.top_values_per_stage:
            raise RuntimeError(
                f"{stage_name} has fewer than two safety-eligible candidates"
            )
        self.state.setdefault("stages", {})[stage_name] = {
            "status": "completed",
            "candidate_count": len(unique_candidates),
            "ranking_path": str(ranking_path),
            "winner": str(top.iloc[0]["trial_id"]),
            "top_two": [str(value) for value in top["trial_id"]],
        }
        self._save_state()
        print(
            f"SEARCH_STAGE_COMPLETE stage={stage_name} "
            f"winner={top.iloc[0]['trial_id']}",
            flush=True,
        )
        return ranking, baseline

    @staticmethod
    def _row_parameters(row: pd.Series) -> ParameterSet:
        return ParameterSet(
            float(row["target_minority_ratio"]),
            float(row["physics_timestep_fraction"]),
            float(row["condition_dropout"]),
        )

    def _write_plan(self) -> Path:
        experiment = load_config(self.config.experiment_config)
        project_count = sum(1 for _ in experiment.targets())
        family_count = len(experiment.data.families)
        sequential_unique = (
            len(self.config.grids.target_minority_ratio)
            + len(self.config.grids.physics_timestep_fraction)
            + len(self.config.grids.condition_dropout)
            - 2
        )
        interaction_combinations = self.config.selection.top_values_per_stage ** 3
        maximum_unique = sequential_unique + interaction_combinations - 1
        screening_tasks = (
            maximum_unique
            * len(self.config.screening_seeds)
            * family_count
            * self.config.source_cv_folds
        )
        confirmation_tasks = (
            len(self.config.confirmation_seeds)
            * family_count
            * self.config.source_cv_folds
        )
        plan = {
            "search_signature": self.config.signature(),
            "selection_protocol": "source_cv",
            "source_projects": project_count,
            "dataset_families": family_count,
            "source_cv_folds": self.config.source_cv_folds,
            "source_cv_validation_fraction": (
                self.config.source_cv_validation_fraction
            ),
            "screening_seed_count": len(self.config.screening_seeds),
            "confirmation_seed_count": len(self.config.confirmation_seeds),
            "sequential_unique_configurations": sequential_unique,
            "interaction_combinations": interaction_combinations,
            "maximum_new_interaction_configurations": interaction_combinations - 1,
            "maximum_unique_configurations": maximum_unique,
            "maximum_screening_tasks": screening_tasks,
            "winner_confirmation_tasks": confirmation_tasks,
            "maximum_total_tasks": screening_tasks + confirmation_tasks,
            "initial_values": self.config.initial.to_dict(),
            "parameter_grids": asdict(self.config.grids),
            "selection_metrics": list(self.config.selection.metrics),
            "maximum_allowed_decline": dict(
                self.config.selection.safety_tolerances
            ),
        }
        path = self.config.output_dir / "search_plan.yaml"
        _atomic_yaml(path, plan)
        return path

    def run(self, *, dry_run: bool = False) -> ParameterSet:
        plan_path = self._write_plan()
        if dry_run:
            print(f"SEARCH_DRY_RUN plan={plan_path}", flush=True)
            print("No training task was started.", flush=True)
            return self.config.initial

        self.state["status"] = "running"
        if self.state.get("started_at") is None:
            self.state["started_at"] = _utc_now()
        self._save_state()
        try:
            stage_one, baseline_metrics = self._run_stage(
                "stage_01_target_minority_ratio",
                stage_one_candidates(self.config),
                None,
            )
            eligible_one = stage_one[stage_one["safety_eligible"]]
            top_target_ratios = tuple(
                float(value)
                for value in eligible_one.head(2)["target_minority_ratio"]
            )
            best_target_ratio = top_target_ratios[0]

            stage_two, _ = self._run_stage(
                "stage_02_physics_timestep_fraction",
                stage_two_candidates(self.config, best_target_ratio),
                baseline_metrics,
            )
            eligible_two = stage_two[stage_two["safety_eligible"]]
            top_physics_timestep_fractions = tuple(
                float(value)
                for value in eligible_two.head(2)["physics_timestep_fraction"]
            )
            best_physics_timestep_fraction = top_physics_timestep_fractions[0]

            stage_three, _ = self._run_stage(
                "stage_03_condition_dropout",
                stage_three_candidates(
                    self.config,
                    best_target_ratio,
                    best_physics_timestep_fraction,
                ),
                baseline_metrics,
            )
            eligible_three = stage_three[stage_three["safety_eligible"]]
            top_condition_dropouts = tuple(
                float(value)
                for value in eligible_three.head(2)["condition_dropout"]
            )

            interactions = interaction_candidates(
                top_target_ratios,
                top_physics_timestep_fractions,
                top_condition_dropouts,
            )
            interaction_ranking, _ = self._run_stage(
                "stage_04_interaction_2x2x2",
                interactions,
                baseline_metrics,
            )
            provisional_row = interaction_ranking[
                interaction_ranking["safety_eligible"]
            ].iloc[0]
            winner = self._row_parameters(provisional_row)
            self.state["provisional_winner"] = winner.to_dict()
            self._save_state()

            self.execute_trial(
                winner,
                self.config.confirmation_seeds,
                "stage_05_winner_confirmation",
            )
            all_seed_metrics = self._source_cv_project_metrics(
                winner, self.config.all_seeds
            )
            final_project_path = (
                self.config.output_dir / "best_source_cv_project_metrics_10_seeds.csv"
            )
            _atomic_csv(final_project_path, all_seed_metrics)
            macro = {
                f"macro_{metric}": float(all_seed_metrics[metric].mean())
                for metric in self.config.selection.metrics
            }
            comparison_png, comparison_pdf = create_parameter_comparison_figure(
                self.config.output_dir,
                winner.to_dict(),
            )
            best_path = self.config.output_dir / "best_params.yaml"
            _atomic_yaml(
                best_path,
                {
                    "method": "Implicit Physics-Guided Diffusion Oversampling (IPGDO)",
                    "status": "completed",
                    "selected_at": _utc_now(),
                    "search_signature": self.config.signature(),
                    "parameters": winner.to_dict(),
                    "fixed_overrides": dict(self.config.fixed_overrides),
                    "screening_seeds": list(self.config.screening_seeds),
                    "confirmation_seeds": list(self.config.confirmation_seeds),
                    "total_winner_seeds": len(self.config.all_seeds),
                    "selection": {
                        "protocol": "source_cv",
                        "folds": self.config.source_cv_folds,
                        "mean_rank_on_screening_seeds": float(
                            provisional_row["mean_rank"]
                        ),
                        "safety_eligible": bool(
                            provisional_row["safety_eligible"]
                        ),
                        **macro,
                    },
                    "apply_as_validation_overrides": {
                        "target_minority_ratio": winner.target_minority_ratio,
                        "diffusion": {
                            "physics_timestep_fraction": (
                                winner.physics_timestep_fraction
                            ),
                            "condition_dropout": winner.condition_dropout,
                        },
                    },
                    "artifacts": {
                        "source_cv_project_metrics": str(final_project_path),
                        "search_state": str(self.state_path),
                        "parameter_comparison_png": str(comparison_png),
                        "parameter_comparison_pdf": str(comparison_pdf),
                    },
                },
            )
            self.state["status"] = "completed"
            self.state["completed_at"] = _utc_now()
            self.state["final_winner"] = winner.to_dict()
            self.state["best_params_path"] = str(best_path)
            self._save_state()
            print(
                f"SEARCH_COMPLETE winner={winner.trial_id} best_params={best_path}",
                flush=True,
            )
            return winner
        except BaseException as exc:
            self.state["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            self.state["last_error"] = f"{type(exc).__name__}: {exc}"
            self._save_state()
            raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Source-CV sequential and local-interaction parameter search for IPGDO"
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--workers",
        type=int,
        help="override execution.workers without changing the YAML file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate configuration and write the task plan without training",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate imports and configuration without writing search outputs",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_parameter_search_config(Path(args.config))
        if args.check:
            print(
                "IPGDO_PARAMETER_SEARCH_OK "
                f"config={config.source_path} screening_seeds={len(config.screening_seeds)} "
                f"confirmation_seeds={len(config.confirmation_seeds)}"
            )
            return 0
        engine = ParameterSearchEngine(config, workers=args.workers)
        engine.run(dry_run=bool(args.dry_run))
        return 0
    except KeyboardInterrupt:
        print("IPGDO parameter search interrupted; rerun to resume.", file=sys.stderr)
        return 130
    except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        print(f"IPGDO parameter search failed: {exc}", file=sys.stderr)
        return 1


__all__ = [
    "ParameterSearchConfig",
    "ParameterSearchEngine",
    "ParameterSet",
    "aggregate_source_cv_project_metrics",
    "interaction_candidates",
    "load_parameter_search_config",
    "main",
    "rank_candidates",
    "stage_one_candidates",
    "stage_three_candidates",
    "stage_two_candidates",
    "unique_parameter_sets",
]
