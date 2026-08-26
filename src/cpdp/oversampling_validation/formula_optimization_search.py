from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import pandas as pd
import yaml

from cpdp.config import load_config
from cpdp.metrics import METRIC_NAMES
from cpdp.oversampling_validation.formula_optimization_plot import (
    create_formula_optimization_figure,
)
from cpdp.oversampling_validation.manifold_formula_search import (
    FinalCPDPConfig,
    ManifoldFormulaSearchEngine,
)
from cpdp.oversampling_validation.parameter_search import (
    ExecutionConfig,
    SelectionConfig,
    _atomic_csv,
    _atomic_yaml,
    _expand_path,
    _mapping,
    _number_slug,
    _utc_now,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "ipgdo_formula_optimization_search.yaml"


def _float_tuple(values: Iterable[Any], name: str) -> Tuple[float, ...]:
    parsed = tuple(float(value) for value in values)
    if len(parsed) < 2:
        raise ValueError(f"{name} must contain at least two values")
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"{name} must not contain duplicate values")
    return parsed


def _int_tuple(values: Iterable[Any], name: str) -> Tuple[int, ...]:
    parsed = tuple(int(value) for value in values)
    if len(parsed) < 2:
        raise ValueError(f"{name} must contain at least two values")
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"{name} must not contain duplicate values")
    return parsed


@dataclass(frozen=True)
class FormulaOptimizationParameterSet:
    formula_l1_c: float
    diffusion_learning_rate: float
    diffusion_epochs: int

    def validate(self) -> None:
        if self.formula_l1_c <= 0:
            raise ValueError("formula_l1_c must be positive")
        if self.diffusion_learning_rate <= 0:
            raise ValueError("diffusion_learning_rate must be positive")
        if self.diffusion_epochs < 1:
            raise ValueError("diffusion_epochs must be positive")

    @property
    def trial_id(self) -> str:
        return (
            f"formula_l1_c_{_number_slug(self.formula_l1_c)}"
            f"__learning_rate_{_number_slug(self.diffusion_learning_rate)}"
            f"__epochs_{self.diffusion_epochs}"
        )

    @property
    def estimated_cost(self) -> float:
        return float(self.diffusion_epochs)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "formula_l1_c": float(self.formula_l1_c),
            "diffusion_learning_rate": float(self.diffusion_learning_rate),
            "diffusion_epochs": int(self.diffusion_epochs),
        }


@dataclass(frozen=True)
class FormulaOptimizationGrids:
    formula_l1_c: Tuple[float, ...]
    diffusion_learning_rate: Tuple[float, ...]
    diffusion_epochs: Tuple[int, ...]


@dataclass(frozen=True)
class FormulaOptimizationSearchConfig:
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
    initial: FormulaOptimizationParameterSet
    grids: FormulaOptimizationGrids
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
        if self.initial.formula_l1_c not in self.grids.formula_l1_c:
            raise ValueError("initial formula_l1_c must occur in its grid")
        if (
            self.initial.diffusion_learning_rate
            not in self.grids.diffusion_learning_rate
        ):
            raise ValueError(
                "initial diffusion_learning_rate must occur in its grid"
            )
        if self.initial.diffusion_epochs not in self.grids.diffusion_epochs:
            raise ValueError("initial diffusion_epochs must occur in its grid")
        if any(value <= 0 for value in self.grids.formula_l1_c):
            raise ValueError("formula_l1_c candidates must be positive")
        if any(value <= 0 for value in self.grids.diffusion_learning_rate):
            raise ValueError("diffusion_learning_rate candidates must be positive")
        if any(value < 1 for value in self.grids.diffusion_epochs):
            raise ValueError("diffusion_epochs candidates must be positive")
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


def load_formula_optimization_search_config(
    path: Path,
) -> FormulaOptimizationSearchConfig:
    source_path = path.resolve()
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping) or raw.get("schema_version") != 1:
        raise ValueError("Unsupported formula-optimization search schema_version")

    project_root = source_path.parent.parent
    paths = _mapping(raw.get("paths"), "paths")
    experiment_config = _expand_path(
        str(paths.get("experiment_config", "experiment.yaml")), source_path.parent
    )
    output_dir = _expand_path(
        str(paths.get("output_dir", "outputs/ipgdo_formula_optimization_search")),
        project_root,
    )
    validation_output_dir = _expand_path(
        str(
            paths.get(
                "validation_output_dir",
                "outputs/ipgdo_formula_optimization_search/validation_runs",
            )
        ),
        project_root,
    )
    final_cpdp_output_dir = _expand_path(
        str(
            paths.get(
                "final_cpdp_output_dir",
                "outputs/ipgdo_formula_optimization_search/final_cpdp_runs",
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
    initial = FormulaOptimizationParameterSet(
        formula_l1_c=float(initial_raw["formula_l1_c"]),
        diffusion_learning_rate=float(initial_raw["diffusion_learning_rate"]),
        diffusion_epochs=int(initial_raw["diffusion_epochs"]),
    )
    grids_raw = _mapping(raw.get("parameter_grids"), "parameter_grids")
    grids = FormulaOptimizationGrids(
        formula_l1_c=_float_tuple(
            grids_raw.get("formula_l1_c", ()),
            "parameter_grids.formula_l1_c",
        ),
        diffusion_learning_rate=_float_tuple(
            grids_raw.get("diffusion_learning_rate", ()),
            "parameter_grids.diffusion_learning_rate",
        ),
        diffusion_epochs=_int_tuple(
            grids_raw.get("diffusion_epochs", ()),
            "parameter_grids.diffusion_epochs",
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

    config = FormulaOptimizationSearchConfig(
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
    config: FormulaOptimizationSearchConfig,
) -> Tuple[FormulaOptimizationParameterSet, ...]:
    return tuple(
        FormulaOptimizationParameterSet(
            value,
            config.initial.diffusion_learning_rate,
            config.initial.diffusion_epochs,
        )
        for value in config.grids.formula_l1_c
    )


def stage_two_candidates(
    config: FormulaOptimizationSearchConfig,
    best_formula_l1_c: float,
) -> Tuple[FormulaOptimizationParameterSet, ...]:
    return tuple(
        FormulaOptimizationParameterSet(
            best_formula_l1_c,
            value,
            config.initial.diffusion_epochs,
        )
        for value in config.grids.diffusion_learning_rate
    )


def stage_three_candidates(
    config: FormulaOptimizationSearchConfig,
    best_formula_l1_c: float,
    best_learning_rate: float,
) -> Tuple[FormulaOptimizationParameterSet, ...]:
    return tuple(
        FormulaOptimizationParameterSet(
            best_formula_l1_c,
            best_learning_rate,
            value,
        )
        for value in config.grids.diffusion_epochs
    )


def interaction_candidates(
    formula_l1_values: Sequence[float],
    learning_rates: Sequence[float],
    epochs: Sequence[int],
) -> Tuple[FormulaOptimizationParameterSet, ...]:
    return tuple(
        FormulaOptimizationParameterSet(formula_l1_c, learning_rate, epoch_count)
        for formula_l1_c in formula_l1_values
        for learning_rate in learning_rates
        for epoch_count in epochs
    )


class FormulaOptimizationSearchEngine(ManifoldFormulaSearchEngine):
    config: FormulaOptimizationSearchConfig

    def _effective_overrides(
        self, parameters: FormulaOptimizationParameterSet
    ) -> Dict[str, Any]:
        overrides = deepcopy(dict(self.config.fixed_overrides))
        formula = dict(_mapping(overrides.get("formula"), "fixed_overrides.formula"))
        formula["l1_c"] = float(parameters.formula_l1_c)
        overrides["formula"] = formula
        diffusion = dict(
            _mapping(overrides.get("diffusion"), "fixed_overrides.diffusion")
        )
        diffusion["learning_rate"] = float(parameters.diffusion_learning_rate)
        diffusion["epochs"] = int(parameters.diffusion_epochs)
        overrides["diffusion"] = diffusion
        return overrides

    @staticmethod
    def _row_parameters(row: pd.Series) -> FormulaOptimizationParameterSet:
        return FormulaOptimizationParameterSet(
            formula_l1_c=float(row["formula_l1_c"]),
            diffusion_learning_rate=float(row["diffusion_learning_rate"]),
            diffusion_epochs=int(row["diffusion_epochs"]),
        )

    def _write_plan(self) -> Path:
        experiment = load_config(self.config.experiment_config)
        project_count = sum(1 for _ in experiment.targets())
        family_count = len(experiment.data.families)
        sequential_unique = (
            len(self.config.grids.formula_l1_c)
            + len(self.config.grids.diffusion_learning_rate)
            + len(self.config.grids.diffusion_epochs)
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

    def run(self, *, dry_run: bool = False) -> FormulaOptimizationParameterSet:
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
                "stage_01_formula_l1_c",
                stage_one_candidates(self.config),
                None,
            )
            eligible_one = stage_one[stage_one["safety_eligible"]].head(2)
            top_formula_l1_values = tuple(
                float(value) for value in eligible_one["formula_l1_c"]
            )
            best_formula_l1_c = top_formula_l1_values[0]

            stage_two, _ = self._run_stage(
                "stage_02_diffusion_learning_rate",
                stage_two_candidates(self.config, best_formula_l1_c),
                baseline_metrics,
            )
            eligible_two = stage_two[stage_two["safety_eligible"]].head(2)
            top_learning_rates = tuple(
                float(value) for value in eligible_two["diffusion_learning_rate"]
            )
            best_learning_rate = top_learning_rates[0]

            stage_three, _ = self._run_stage(
                "stage_03_diffusion_epochs",
                stage_three_candidates(
                    self.config,
                    best_formula_l1_c,
                    best_learning_rate,
                ),
                baseline_metrics,
            )
            eligible_three = stage_three[stage_three["safety_eligible"]].head(2)
            top_epochs = tuple(int(value) for value in eligible_three["diffusion_epochs"])

            interactions = interaction_candidates(
                top_formula_l1_values,
                top_learning_rates,
                top_epochs,
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
            comparison_png, comparison_pdf = create_formula_optimization_figure(
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
                        "safety_eligible": bool(provisional_row["safety_eligible"]),
                        **source_cv_macro,
                    },
                    "apply_as_validation_overrides": {
                        "formula": {"l1_c": winner.formula_l1_c},
                        "diffusion": {
                            "learning_rate": winner.diffusion_learning_rate,
                            "epochs": winner.diffusion_epochs,
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
            "Search IPGDO formula/optimizer parameters and run the 10-seed "
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
        config = load_formula_optimization_search_config(Path(args.config))
        if args.check:
            print(
                "IPGDO_FORMULA_OPTIMIZATION_SEARCH_OK "
                f"config={config.source_path} "
                f"screening_seeds={len(config.screening_seeds)} "
                f"confirmation_seeds={len(config.confirmation_seeds)} "
                f"final_cpdp_seeds={len(config.final_cpdp.seeds)}"
            )
            return 0
        engine = FormulaOptimizationSearchEngine(config, workers=args.workers)
        engine.run(dry_run=bool(args.dry_run))
        return 0
    except KeyboardInterrupt:
        print("IPGDO parameter search interrupted; rerun to resume.", file=sys.stderr)
        return 130
    except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        print(f"IPGDO parameter search failed: {exc}", file=sys.stderr)
        return 1


__all__ = [
    "FormulaOptimizationGrids",
    "FormulaOptimizationParameterSet",
    "FormulaOptimizationSearchConfig",
    "FormulaOptimizationSearchEngine",
    "interaction_candidates",
    "load_formula_optimization_search_config",
    "main",
    "stage_one_candidates",
    "stage_three_candidates",
    "stage_two_candidates",
]
