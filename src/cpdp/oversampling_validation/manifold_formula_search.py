from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

from cpdp.config import load_config
from cpdp.metrics import METRIC_NAMES
from cpdp.oversampling_validation.config import load_validation_config
from cpdp.oversampling_validation.manifold_formula_plot import (
    create_manifold_formula_figure,
)
from cpdp.oversampling_validation.parameter_search import (
    ExecutionConfig,
    ParameterSearchEngine,
    SelectionConfig,
    _atomic_csv,
    _atomic_yaml,
    _expand_path,
    _mapping,
    _number_slug,
    _utc_now,
    unique_parameter_sets,
)
from cpdp.oversampling_validation.runner import OversamplingValidationRunner
from cpdp.oversampling_validation.store import ValidationStore


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "ipgdo_manifold_formula_search.yaml"


def _int_tuple(values: Iterable[Any], name: str) -> Tuple[int, ...]:
    parsed = tuple(int(value) for value in values)
    if len(parsed) < 2:
        raise ValueError(f"{name} must contain at least two values")
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"{name} must not contain duplicate values")
    return parsed


@dataclass(frozen=True)
class ManifoldInterval:
    lower_quantile: float
    upper_quantile: float

    def validate(self) -> None:
        if not 0.0 <= self.lower_quantile < self.upper_quantile <= 1.0:
            raise ValueError(
                "manifold interval must satisfy 0 <= lower_quantile "
                "< upper_quantile <= 1"
            )

    def to_dict(self) -> Dict[str, float]:
        return {
            "lower_quantile": float(self.lower_quantile),
            "upper_quantile": float(self.upper_quantile),
        }


@dataclass(frozen=True)
class ManifoldFormulaParameterSet:
    manifold_lower_quantile: float
    manifold_upper_quantile: float
    manifold_neighbors: int
    formula_max_terms: int

    @property
    def interval(self) -> ManifoldInterval:
        return ManifoldInterval(
            self.manifold_lower_quantile,
            self.manifold_upper_quantile,
        )

    def validate(self) -> None:
        self.interval.validate()
        if self.manifold_neighbors < 1:
            raise ValueError("manifold_neighbors must be positive")
        if self.formula_max_terms < 1:
            raise ValueError("formula_max_terms must be positive")

    @property
    def trial_id(self) -> str:
        return (
            f"manifold_{_number_slug(self.manifold_lower_quantile)}"
            f"_{_number_slug(self.manifold_upper_quantile)}"
            f"__neighbors_{self.manifold_neighbors}"
            f"__formula_terms_{self.formula_max_terms}"
        )

    @property
    def estimated_cost(self) -> float:
        return float(self.formula_max_terms + self.manifold_neighbors / 100.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "manifold_lower_quantile": float(self.manifold_lower_quantile),
            "manifold_upper_quantile": float(self.manifold_upper_quantile),
            "manifold_neighbors": int(self.manifold_neighbors),
            "formula_max_terms": int(self.formula_max_terms),
        }


@dataclass(frozen=True)
class ManifoldFormulaGrids:
    manifold_intervals: Tuple[ManifoldInterval, ...]
    manifold_neighbors: Tuple[int, ...]
    formula_max_terms: Tuple[int, ...]


@dataclass(frozen=True)
class FinalCPDPConfig:
    enabled: bool
    output_dir: Path
    profile_name: str
    seeds: Tuple[int, ...]
    validation_fraction: float

    def validate(self) -> None:
        if not self.enabled:
            raise ValueError("final_cpdp.enabled must remain true for this search")
        if len(self.seeds) != 10 or len(set(self.seeds)) != 10:
            raise ValueError("final_cpdp must use exactly 10 unique seeds")
        if not 0.0 < self.validation_fraction < 0.5:
            raise ValueError("final_cpdp.validation_fraction must be in (0, 0.5)")


@dataclass(frozen=True)
class ManifoldFormulaSearchConfig:
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
    initial: ManifoldFormulaParameterSet
    grids: ManifoldFormulaGrids
    screening_seeds: Tuple[int, ...]
    confirmation_seeds: Tuple[int, ...]
    fixed_overrides: Mapping[str, Any]
    source_cv_folds: int
    source_cv_validation_fraction: float
    classifier: Mapping[str, Any]
    quality: Mapping[str, Any]
    selection: SelectionConfig
    execution: ExecutionConfig
    final_cpdp: FinalCPDPConfig

    @property
    def all_seeds(self) -> Tuple[int, ...]:
        return self.screening_seeds + self.confirmation_seeds

    def validate(self) -> None:
        self.initial.validate()
        if len(self.grids.manifold_intervals) < 2:
            raise ValueError("parameter_grids.manifold_intervals needs at least two values")
        for interval in self.grids.manifold_intervals:
            interval.validate()
        if len(set(self.grids.manifold_intervals)) != len(
            self.grids.manifold_intervals
        ):
            raise ValueError("parameter_grids.manifold_intervals contains duplicates")
        if self.initial.interval not in self.grids.manifold_intervals:
            raise ValueError("initial manifold interval must occur in its grid")
        if self.initial.manifold_neighbors not in self.grids.manifold_neighbors:
            raise ValueError("initial manifold_neighbors must occur in its grid")
        if self.initial.formula_max_terms not in self.grids.formula_max_terms:
            raise ValueError("initial formula_max_terms must occur in its grid")
        for value in self.grids.manifold_neighbors:
            if value < 1:
                raise ValueError("manifold_neighbors candidates must be positive")
        for value in self.grids.formula_max_terms:
            if value < 1:
                raise ValueError("formula_max_terms candidates must be positive")
        if self.source_cv_folds < 2:
            raise ValueError("validation.source_cv.folds must be at least 2")
        if not 0.0 < self.source_cv_validation_fraction < 0.5:
            raise ValueError(
                "validation.source_cv.validation_fraction must be in (0, 0.5)"
            )
        if not self.screening_seeds or not self.confirmation_seeds:
            raise ValueError("screening and confirmation seeds cannot be empty")
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
        self.final_cpdp.validate()
        if self.final_cpdp.seeds != self.all_seeds:
            raise ValueError("final_cpdp seeds must equal all 10 search seeds")

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
            "final_cpdp": asdict(self.final_cpdp),
        }
        encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]


def _parse_interval(value: Any, name: str) -> ManifoldInterval:
    mapping = _mapping(value, name)
    interval = ManifoldInterval(
        lower_quantile=float(mapping["lower_quantile"]),
        upper_quantile=float(mapping["upper_quantile"]),
    )
    interval.validate()
    return interval


def load_manifold_formula_search_config(path: Path) -> ManifoldFormulaSearchConfig:
    source_path = path.resolve()
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping) or raw.get("schema_version") != 1:
        raise ValueError("Unsupported manifold-formula search schema_version")

    project_root = source_path.parent.parent
    paths = _mapping(raw.get("paths"), "paths")
    experiment_config = _expand_path(
        str(paths.get("experiment_config", "experiment.yaml")), source_path.parent
    )
    output_dir = _expand_path(
        str(paths.get("output_dir", "outputs/ipgdo_manifold_formula_search")),
        project_root,
    )
    validation_output_dir = _expand_path(
        str(
            paths.get(
                "validation_output_dir",
                "outputs/ipgdo_manifold_formula_search/validation_runs",
            )
        ),
        project_root,
    )
    final_cpdp_output_dir = _expand_path(
        str(
            paths.get(
                "final_cpdp_output_dir",
                "outputs/ipgdo_manifold_formula_search/final_cpdp_runs",
            )
        ),
        project_root,
    )
    runner_script = _expand_path(
        str(paths.get("runner_script", "run_oversampling_validation.py")),
        project_root,
    )

    method = _mapping(raw.get("method"), "method")
    initial_raw = _mapping(raw.get("initial_values"), "initial_values")
    initial_interval = _parse_interval(
        initial_raw.get("manifold_interval"),
        "initial_values.manifold_interval",
    )
    initial = ManifoldFormulaParameterSet(
        manifold_lower_quantile=initial_interval.lower_quantile,
        manifold_upper_quantile=initial_interval.upper_quantile,
        manifold_neighbors=int(initial_raw["manifold_neighbors"]),
        formula_max_terms=int(initial_raw["formula_max_terms"]),
    )
    grids_raw = _mapping(raw.get("parameter_grids"), "parameter_grids")
    intervals = tuple(
        _parse_interval(value, f"parameter_grids.manifold_intervals[{index}]")
        for index, value in enumerate(grids_raw.get("manifold_intervals", ()))
    )
    grids = ManifoldFormulaGrids(
        manifold_intervals=intervals,
        manifold_neighbors=_int_tuple(
            grids_raw.get("manifold_neighbors", ()),
            "parameter_grids.manifold_neighbors",
        ),
        formula_max_terms=_int_tuple(
            grids_raw.get("formula_max_terms", ()),
            "parameter_grids.formula_max_terms",
        ),
    )

    seeds_raw = _mapping(raw.get("seeds"), "seeds")
    screening_seeds = tuple(int(seed) for seed in seeds_raw.get("screening", ()))
    confirmation_seeds = tuple(
        int(seed) for seed in seeds_raw.get("confirmation", ())
    )
    all_seeds = screening_seeds + confirmation_seeds

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
        top_values_per_stage=int(selection_raw.get("top_values_per_stage", 2)),
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
    final_raw = _mapping(raw.get("final_cpdp"), "final_cpdp")
    if not bool(final_raw.get("use_all_search_seeds", True)):
        raise ValueError("final_cpdp.use_all_search_seeds must be true")
    final_cpdp = FinalCPDPConfig(
        enabled=bool(final_raw.get("enabled", True)),
        output_dir=final_cpdp_output_dir,
        profile_name=str(method.get("final_cpdp_profile_name", "final_cpdp")),
        seeds=all_seeds,
        validation_fraction=float(final_raw.get("validation_fraction", 0.2)),
    )

    config = ManifoldFormulaSearchConfig(
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
        fixed_overrides=dict(_mapping(raw.get("fixed_overrides"), "fixed_overrides")),
        source_cv_folds=int(source_cv.get("folds", 3)),
        source_cv_validation_fraction=float(
            source_cv.get("validation_fraction", 0.2)
        ),
        classifier=dict(_mapping(validation.get("classifier"), "validation.classifier")),
        quality=dict(_mapping(validation.get("quality"), "validation.quality")),
        selection=selection,
        execution=execution,
        final_cpdp=final_cpdp,
    )
    config.validate()
    if not config.experiment_config.is_file():
        raise FileNotFoundError(
            f"Experiment configuration not found: {config.experiment_config}"
        )
    if not config.runner_script.is_file():
        raise FileNotFoundError(f"Validation runner not found: {config.runner_script}")
    return config


def stage_one_candidates(
    config: ManifoldFormulaSearchConfig,
) -> Tuple[ManifoldFormulaParameterSet, ...]:
    return tuple(
        ManifoldFormulaParameterSet(
            interval.lower_quantile,
            interval.upper_quantile,
            config.initial.manifold_neighbors,
            config.initial.formula_max_terms,
        )
        for interval in config.grids.manifold_intervals
    )


def stage_two_candidates(
    config: ManifoldFormulaSearchConfig,
    best_interval: ManifoldInterval,
) -> Tuple[ManifoldFormulaParameterSet, ...]:
    return tuple(
        ManifoldFormulaParameterSet(
            best_interval.lower_quantile,
            best_interval.upper_quantile,
            neighbors,
            config.initial.formula_max_terms,
        )
        for neighbors in config.grids.manifold_neighbors
    )


def stage_three_candidates(
    config: ManifoldFormulaSearchConfig,
    best_interval: ManifoldInterval,
    best_neighbors: int,
) -> Tuple[ManifoldFormulaParameterSet, ...]:
    return tuple(
        ManifoldFormulaParameterSet(
            best_interval.lower_quantile,
            best_interval.upper_quantile,
            best_neighbors,
            terms,
        )
        for terms in config.grids.formula_max_terms
    )


def interaction_candidates(
    intervals: Sequence[ManifoldInterval],
    neighbors: Sequence[int],
    formula_terms: Sequence[int],
) -> Tuple[ManifoldFormulaParameterSet, ...]:
    return tuple(
        ManifoldFormulaParameterSet(
            interval.lower_quantile,
            interval.upper_quantile,
            neighbor_count,
            max_terms,
        )
        for interval in intervals
        for neighbor_count in neighbors
        for max_terms in formula_terms
    )


def aggregate_cpdp_target_metrics(
    frame: pd.DataFrame,
    *,
    variant: str,
    seeds: Sequence[int],
    targets: Sequence[Tuple[str, str]],
    metrics: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    metric_columns = [f"augmented_{metric}" for metric in metrics]
    required = {"protocol", "variant", "family", "target", "seed", *metric_columns}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"CPDP summary is missing: {', '.join(missing)}")
    selected = frame[
        (frame["protocol"] == "cpdp")
        & (frame["variant"] == variant)
        & (frame["seed"].isin([int(seed) for seed in seeds]))
    ].copy()
    per_seed = (
        selected.groupby(["family", "target", "seed"], as_index=False)[metric_columns]
        .mean(numeric_only=True)
    )
    expected_targets = {(str(family), str(target)) for family, target in targets}
    actual_targets = set(
        zip(per_seed["family"].astype(str), per_seed["target"].astype(str))
    )
    expected_rows = len(expected_targets) * len(seeds)
    if actual_targets != expected_targets or len(per_seed) != expected_rows:
        raise RuntimeError(
            f"Final CPDP has {len(per_seed)}/{expected_rows} target-seed rows"
        )
    counts = per_seed.groupby(["family", "target"])[metric_columns].count()
    if not counts.eq(len(seeds)).all().all():
        raise RuntimeError("Final CPDP contains missing metric values")

    rename_metrics = {
        f"augmented_{metric}": metric for metric in metrics
    }
    run_metrics = per_seed.rename(
        columns={"family": "dataset", "target": "project", **rename_metrics}
    )
    target_means = (
        per_seed.groupby(["family", "target"], as_index=False)[metric_columns]
        .mean(numeric_only=True)
        .rename(
            columns={"family": "dataset", "target": "project", **rename_metrics}
        )
    )
    target_means.insert(2, "seed_count", len(seeds))
    order = {
        (str(family), str(target)): index
        for index, (family, target) in enumerate(targets)
    }
    run_metrics["__order"] = [
        order[(str(dataset), str(project))]
        for dataset, project in zip(run_metrics["dataset"], run_metrics["project"])
    ]
    target_means["__order"] = [
        order[(str(dataset), str(project))]
        for dataset, project in zip(target_means["dataset"], target_means["project"])
    ]
    run_metrics = run_metrics.sort_values(["__order", "seed"]).drop(
        columns="__order"
    )
    target_means = target_means.sort_values("__order").drop(columns="__order")
    macro = {
        f"macro_{metric}": float(target_means[metric].mean()) for metric in metrics
    }
    return run_metrics.reset_index(drop=True), target_means.reset_index(drop=True), macro


class ManifoldFormulaSearchEngine(ParameterSearchEngine):
    config: ManifoldFormulaSearchConfig

    def _effective_overrides(
        self, parameters: ManifoldFormulaParameterSet
    ) -> Dict[str, Any]:
        overrides = deepcopy(dict(self.config.fixed_overrides))
        formula = dict(_mapping(overrides.get("formula"), "fixed_overrides.formula"))
        formula["max_terms"] = int(parameters.formula_max_terms)
        overrides["formula"] = formula
        manifold = dict(
            _mapping(overrides.get("manifold"), "fixed_overrides.manifold")
        )
        manifold.update(
            {
                "lower_quantile": float(parameters.manifold_lower_quantile),
                "upper_quantile": float(parameters.manifold_upper_quantile),
                "neighbors": int(parameters.manifold_neighbors),
            }
        )
        overrides["manifold"] = manifold
        return overrides

    @staticmethod
    def _row_parameters(row: pd.Series) -> ManifoldFormulaParameterSet:
        return ManifoldFormulaParameterSet(
            manifold_lower_quantile=float(row["manifold_lower_quantile"]),
            manifold_upper_quantile=float(row["manifold_upper_quantile"]),
            manifold_neighbors=int(row["manifold_neighbors"]),
            formula_max_terms=int(row["formula_max_terms"]),
        )

    def _write_plan(self) -> Path:
        experiment = load_config(self.config.experiment_config)
        project_count = sum(1 for _ in experiment.targets())
        family_count = len(experiment.data.families)
        sequential_unique = (
            len(self.config.grids.manifold_intervals)
            + len(self.config.grids.manifold_neighbors)
            + len(self.config.grids.formula_max_terms)
            - 2
        )
        interaction_count = self.config.selection.top_values_per_stage**3
        maximum_unique = sequential_unique + interaction_count - 1
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
        final_cpdp_tasks = project_count * len(self.config.final_cpdp.seeds)
        plan = {
            "search_signature": self.config.signature(),
            "selection_protocol": "source_cv",
            "source_projects": project_count,
            "source_families": family_count,
            "source_cv_folds": self.config.source_cv_folds,
            "maximum_unique_configurations": maximum_unique,
            "maximum_screening_tasks": screening_tasks,
            "winner_confirmation_tasks": confirmation_tasks,
            "final_cpdp_tasks": final_cpdp_tasks,
            "maximum_total_tasks": (
                screening_tasks + confirmation_tasks + final_cpdp_tasks
            ),
            "initial_values": self.config.initial.to_dict(),
            "parameter_grids": asdict(self.config.grids),
            "fixed_overrides": dict(self.config.fixed_overrides),
            "selection_metrics": list(self.config.selection.metrics),
            "maximum_allowed_decline": dict(
                self.config.selection.safety_tolerances
            ),
        }
        path = self.config.output_dir / "search_plan.yaml"
        _atomic_yaml(path, plan)
        return path

    def _final_cpdp_mapping(
        self, parameters: ManifoldFormulaParameterSet
    ) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "experiment_config": str(self.config.experiment_config),
            "output_dir": str(self.config.final_cpdp.output_dir),
            "oversampling_method": self.config.oversampling_method,
            "variants": {
                self.config.variant_name: {
                    "kind": self.config.variant_kind,
                    "overrides": self._effective_overrides(parameters),
                }
            },
            "profiles": {
                self.config.final_cpdp.profile_name: {
                    "oversampling_profile": self.config.oversampling_profile,
                    "seeds": list(self.config.final_cpdp.seeds),
                    "variants": [self.config.variant_name],
                    "source_cv": {
                        "enabled": False,
                        "folds": self.config.source_cv_folds,
                        "validation_fraction": self.config.source_cv_validation_fraction,
                    },
                    "cpdp": {
                        "enabled": True,
                        "validation_fraction": (
                            self.config.final_cpdp.validation_fraction
                        ),
                    },
                    "classifier": dict(self.config.classifier),
                    "quality": dict(self.config.quality),
                }
            },
        }

    def _run_final_cpdp(
        self, parameters: ManifoldFormulaParameterSet
    ) -> Dict[str, Any]:
        profile_name = self.config.final_cpdp.profile_name
        config_path = self.generated_config_dir / (
            f"{parameters.trial_id}__final_cpdp__code_{self.source_signature}.yaml"
        )
        _atomic_yaml(config_path, self._final_cpdp_mapping(parameters))
        validation_config = load_validation_config(config_path)
        runner = OversamplingValidationRunner(validation_config)
        fingerprint = runner.fingerprint(profile_name)
        store = ValidationStore(
            validation_config.output_dir,
            fingerprint,
            profile_name,
        )
        targets = tuple(validation_config.experiment.targets())
        seeds = self.config.final_cpdp.seeds
        expected_tasks = len(targets) * len(seeds)

        def completed_tasks() -> int:
            return sum(
                store.is_complete(
                    store.run_path(
                        "cpdp",
                        self.config.variant_name,
                        family,
                        int(seed),
                        target=target,
                    )
                )
                for family, target in targets
                for seed in seeds
            )

        completed = completed_tasks()
        self.state["final_cpdp"] = {
            "status": "reused" if completed == expected_tasks else "running",
            "parameters": parameters.to_dict(),
            "config_path": str(config_path),
            "fingerprint": fingerprint,
            "output_root": str(store.root),
            "expected_tasks": expected_tasks,
            "completed_tasks": completed,
        }
        self._save_state()
        if completed != expected_tasks:
            print(
                f"FINAL_CPDP_RUN winner={parameters.trial_id} "
                f"completed={completed}/{expected_tasks} workers={self.workers}",
                flush=True,
            )
            command = [
                str(self.python_executable),
                str(self.config.runner_script),
                "--config",
                str(config_path),
                "--workers",
                str(self.workers),
                "run",
                "--profile",
                profile_name,
                "--protocol",
                "cpdp",
                "--variant",
                self.config.variant_name,
            ]
            if self.config.execution.fail_fast:
                command.append("--fail-fast")
            completed_process = subprocess.run(
                command,
                cwd=self.config.project_root,
                env=self._subprocess_environment(),
                check=False,
            )
            completed = completed_tasks()
            if completed_process.returncode != 0 or completed != expected_tasks:
                self.state["final_cpdp"].update(
                    {
                        "status": "failed",
                        "completed_tasks": completed,
                        "return_code": completed_process.returncode,
                    }
                )
                self._save_state()
                raise RuntimeError(
                    "Final CPDP is incomplete: "
                    f"return_code={completed_process.returncode}, "
                    f"completed={completed}/{expected_tasks}"
                )
        else:
            print(
                f"FINAL_CPDP_REUSE winner={parameters.trial_id} "
                f"tasks={expected_tasks}",
                flush=True,
            )

        summary_paths = store.summarize()
        utility = pd.read_csv(summary_paths["utility_runs"])
        runs, target_means, macro = aggregate_cpdp_target_metrics(
            utility,
            variant=self.config.variant_name,
            seeds=seeds,
            targets=targets,
            metrics=self.config.selection.metrics,
        )
        run_path = self.config.output_dir / "best_cpdp_runs_10_seeds.csv"
        target_path = self.config.output_dir / "best_cpdp_target_means_10_seeds.csv"
        _atomic_csv(run_path, runs)
        _atomic_csv(target_path, target_means)
        self.state["final_cpdp"].update(
            {
                "status": "completed",
                "completed_tasks": expected_tasks,
                "run_metrics_path": str(run_path),
                "target_means_path": str(target_path),
                "macro_metrics": macro,
            }
        )
        self._save_state()
        print(
            f"FINAL_CPDP_COMPLETE tasks={expected_tasks} target_means={target_path}",
            flush=True,
        )
        return {
            "config": str(config_path),
            "fingerprint": fingerprint,
            "output_root": str(store.root),
            "successful_tasks": expected_tasks,
            "seed_count": len(seeds),
            "target_count": len(targets),
            "run_metrics": str(run_path),
            "target_means": str(target_path),
            "macro_metrics": macro,
            "validation_summary": {
                name: str(path) for name, path in summary_paths.items()
            },
        }

    def run(self, *, dry_run: bool = False) -> ManifoldFormulaParameterSet:
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
                "stage_01_manifold_interval",
                stage_one_candidates(self.config),
                None,
            )
            eligible_one = stage_one[stage_one["safety_eligible"]].head(2)
            top_intervals = tuple(
                ManifoldInterval(
                    float(row.manifold_lower_quantile),
                    float(row.manifold_upper_quantile),
                )
                for row in eligible_one.itertuples()
            )
            best_interval = top_intervals[0]

            stage_two, _ = self._run_stage(
                "stage_02_manifold_neighbors",
                stage_two_candidates(self.config, best_interval),
                baseline_metrics,
            )
            eligible_two = stage_two[stage_two["safety_eligible"]].head(2)
            top_neighbors = tuple(int(value) for value in eligible_two["manifold_neighbors"])
            best_neighbors = top_neighbors[0]

            stage_three, _ = self._run_stage(
                "stage_03_formula_max_terms",
                stage_three_candidates(self.config, best_interval, best_neighbors),
                baseline_metrics,
            )
            eligible_three = stage_three[stage_three["safety_eligible"]].head(2)
            top_formula_terms = tuple(
                int(value) for value in eligible_three["formula_max_terms"]
            )

            interactions = interaction_candidates(
                top_intervals,
                top_neighbors,
                top_formula_terms,
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
                winner,
                self.config.all_seeds,
            )
            final_project_path = (
                self.config.output_dir
                / "best_source_cv_project_metrics_10_seeds.csv"
            )
            _atomic_csv(final_project_path, all_seed_metrics)
            source_cv_macro = {
                f"macro_{metric}": float(all_seed_metrics[metric].mean())
                for metric in self.config.selection.metrics
            }
            comparison_png, comparison_pdf = create_manifold_formula_figure(
                self.config.output_dir,
                winner.to_dict(),
            )

            self.state["status"] = "final_cpdp_running"
            self._save_state()
            final_cpdp = self._run_final_cpdp(winner)

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
                    "total_winner_source_cv_seeds": len(self.config.all_seeds),
                    "selection": {
                        "protocol": "source_cv",
                        "folds": self.config.source_cv_folds,
                        "mean_rank_on_screening_seeds": float(
                            provisional_row["mean_rank"]
                        ),
                        "safety_eligible": bool(
                            provisional_row["safety_eligible"]
                        ),
                        **source_cv_macro,
                    },
                    "apply_as_validation_overrides": {
                        "formula": {"max_terms": winner.formula_max_terms},
                        "manifold": {
                            "lower_quantile": winner.manifold_lower_quantile,
                            "upper_quantile": winner.manifold_upper_quantile,
                            "neighbors": winner.manifold_neighbors,
                        },
                    },
                    "final_cpdp": final_cpdp,
                    "artifacts": {
                        "source_cv_project_metrics": str(final_project_path),
                        "search_state": str(self.state_path),
                        "parameter_comparison_png": str(comparison_png),
                        "parameter_comparison_pdf": str(comparison_pdf),
                        "cpdp_run_metrics": final_cpdp["run_metrics"],
                        "cpdp_target_means": final_cpdp["target_means"],
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
            self.state["status"] = (
                "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            )
            self.state["last_error"] = f"{type(exc).__name__}: {exc}"
            self._save_state()
            raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Search IPGDO manifold/formula parameters and run the 10-seed "
            "winner CPDP experiment"
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
        help="validate imports and configuration without writing outputs",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_manifold_formula_search_config(Path(args.config))
        if args.check:
            print(
                "IPGDO_MANIFOLD_FORMULA_SEARCH_OK "
                f"config={config.source_path} "
                f"screening_seeds={len(config.screening_seeds)} "
                f"confirmation_seeds={len(config.confirmation_seeds)} "
                f"final_cpdp_seeds={len(config.final_cpdp.seeds)}"
            )
            return 0
        engine = ManifoldFormulaSearchEngine(config, workers=args.workers)
        engine.run(dry_run=bool(args.dry_run))
        return 0
    except KeyboardInterrupt:
        print("IPGDO parameter search interrupted; rerun to resume.", file=sys.stderr)
        return 130
    except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        print(f"IPGDO parameter search failed: {exc}", file=sys.stderr)
        return 1


__all__ = [
    "ManifoldFormulaParameterSet",
    "ManifoldFormulaSearchConfig",
    "ManifoldFormulaSearchEngine",
    "ManifoldInterval",
    "aggregate_cpdp_target_metrics",
    "interaction_candidates",
    "load_manifold_formula_search_config",
    "main",
    "stage_one_candidates",
    "stage_three_candidates",
    "stage_two_candidates",
]
