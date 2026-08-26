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
from cpdp.oversampling_validation.manifold_formula_search import (
    FinalCPDPConfig,
    ManifoldFormulaSearchEngine,
)
from cpdp.oversampling_validation.network_capacity_plot import (
    create_network_capacity_figure,
)
from cpdp.oversampling_validation.parameter_search import (
    ExecutionConfig,
    SelectionConfig,
    _atomic_csv,
    _atomic_yaml,
    _expand_path,
    _mapping,
    _utc_now,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "ipgdo_network_capacity_search.yaml"


def _int_tuple(values: Iterable[Any], name: str) -> Tuple[int, ...]:
    parsed = tuple(int(value) for value in values)
    if len(parsed) < 2:
        raise ValueError(f"{name} must contain at least two values")
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"{name} must not contain duplicate values")
    return parsed


@dataclass(frozen=True)
class NetworkCapacityParameterSet:
    diffusion_hidden_dim: int
    diffusion_residual_blocks: int
    diffusion_timesteps: int

    def validate(self) -> None:
        if self.diffusion_hidden_dim < 16:
            raise ValueError("diffusion_hidden_dim must be at least 16")
        if self.diffusion_residual_blocks < 1:
            raise ValueError("diffusion_residual_blocks must be positive")
        if self.diffusion_timesteps < 2:
            raise ValueError("diffusion_timesteps must be at least 2")

    @property
    def trial_id(self) -> str:
        return (
            f"hidden_dim_{self.diffusion_hidden_dim}"
            f"__residual_blocks_{self.diffusion_residual_blocks}"
            f"__timesteps_{self.diffusion_timesteps}"
        )

    @property
    def estimated_cost(self) -> float:
        return float(
            self.diffusion_hidden_dim**2
            * self.diffusion_residual_blocks
            * self.diffusion_timesteps
        )

    def to_dict(self) -> Dict[str, int]:
        return {
            "diffusion_hidden_dim": int(self.diffusion_hidden_dim),
            "diffusion_residual_blocks": int(self.diffusion_residual_blocks),
            "diffusion_timesteps": int(self.diffusion_timesteps),
        }


@dataclass(frozen=True)
class NetworkCapacityGrids:
    diffusion_hidden_dim: Tuple[int, ...]
    diffusion_residual_blocks: Tuple[int, ...]
    diffusion_timesteps: Tuple[int, ...]


@dataclass(frozen=True)
class NetworkCapacitySearchConfig:
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
    initial: NetworkCapacityParameterSet
    grids: NetworkCapacityGrids
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
        if self.initial.diffusion_hidden_dim not in self.grids.diffusion_hidden_dim:
            raise ValueError("initial diffusion_hidden_dim must occur in its grid")
        if (
            self.initial.diffusion_residual_blocks
            not in self.grids.diffusion_residual_blocks
        ):
            raise ValueError(
                "initial diffusion_residual_blocks must occur in its grid"
            )
        if self.initial.diffusion_timesteps not in self.grids.diffusion_timesteps:
            raise ValueError("initial diffusion_timesteps must occur in its grid")
        if any(value < 16 for value in self.grids.diffusion_hidden_dim):
            raise ValueError("diffusion_hidden_dim candidates must be at least 16")
        if any(value < 1 for value in self.grids.diffusion_residual_blocks):
            raise ValueError("diffusion_residual_blocks candidates must be positive")
        if any(value < 2 for value in self.grids.diffusion_timesteps):
            raise ValueError("diffusion_timesteps candidates must be at least 2")
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


def load_network_capacity_search_config(path: Path) -> NetworkCapacitySearchConfig:
    source_path = path.resolve()
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping) or raw.get("schema_version") != 1:
        raise ValueError("Unsupported network-capacity search schema_version")

    project_root = source_path.parent.parent
    paths = _mapping(raw.get("paths"), "paths")
    experiment_config = _expand_path(
        str(paths.get("experiment_config", "experiment.yaml")), source_path.parent
    )
    output_dir = _expand_path(
        str(paths.get("output_dir", "outputs/ipgdo_network_capacity_search")),
        project_root,
    )
    validation_output_dir = _expand_path(
        str(
            paths.get(
                "validation_output_dir",
                "outputs/ipgdo_network_capacity_search/validation_runs",
            )
        ),
        project_root,
    )
    final_cpdp_output_dir = _expand_path(
        str(
            paths.get(
                "final_cpdp_output_dir",
                "outputs/ipgdo_network_capacity_search/final_cpdp_runs",
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
    initial = NetworkCapacityParameterSet(
        diffusion_hidden_dim=int(initial_raw["diffusion_hidden_dim"]),
        diffusion_residual_blocks=int(initial_raw["diffusion_residual_blocks"]),
        diffusion_timesteps=int(initial_raw["diffusion_timesteps"]),
    )
    grids_raw = _mapping(raw.get("parameter_grids"), "parameter_grids")
    grids = NetworkCapacityGrids(
        diffusion_hidden_dim=_int_tuple(
            grids_raw.get("diffusion_hidden_dim", ()),
            "parameter_grids.diffusion_hidden_dim",
        ),
        diffusion_residual_blocks=_int_tuple(
            grids_raw.get("diffusion_residual_blocks", ()),
            "parameter_grids.diffusion_residual_blocks",
        ),
        diffusion_timesteps=_int_tuple(
            grids_raw.get("diffusion_timesteps", ()),
            "parameter_grids.diffusion_timesteps",
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

    config = NetworkCapacitySearchConfig(
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
    config: NetworkCapacitySearchConfig,
) -> Tuple[NetworkCapacityParameterSet, ...]:
    return tuple(
        NetworkCapacityParameterSet(
            value,
            config.initial.diffusion_residual_blocks,
            config.initial.diffusion_timesteps,
        )
        for value in config.grids.diffusion_hidden_dim
    )


def stage_two_candidates(
    config: NetworkCapacitySearchConfig,
    best_hidden_dim: int,
) -> Tuple[NetworkCapacityParameterSet, ...]:
    return tuple(
        NetworkCapacityParameterSet(
            best_hidden_dim,
            value,
            config.initial.diffusion_timesteps,
        )
        for value in config.grids.diffusion_residual_blocks
    )


def stage_three_candidates(
    config: NetworkCapacitySearchConfig,
    best_hidden_dim: int,
    best_residual_blocks: int,
) -> Tuple[NetworkCapacityParameterSet, ...]:
    return tuple(
        NetworkCapacityParameterSet(
            best_hidden_dim,
            best_residual_blocks,
            value,
        )
        for value in config.grids.diffusion_timesteps
    )


def interaction_candidates(
    hidden_dims: Sequence[int],
    residual_blocks: Sequence[int],
    timesteps: Sequence[int],
) -> Tuple[NetworkCapacityParameterSet, ...]:
    return tuple(
        NetworkCapacityParameterSet(hidden_dim, block_count, timestep_count)
        for hidden_dim in hidden_dims
        for block_count in residual_blocks
        for timestep_count in timesteps
    )


class NetworkCapacitySearchEngine(ManifoldFormulaSearchEngine):
    config: NetworkCapacitySearchConfig

    def _effective_overrides(
        self, parameters: NetworkCapacityParameterSet
    ) -> Dict[str, Any]:
        overrides = deepcopy(dict(self.config.fixed_overrides))
        diffusion = dict(
            _mapping(overrides.get("diffusion"), "fixed_overrides.diffusion")
        )
        diffusion.update(
            {
                "hidden_dim": int(parameters.diffusion_hidden_dim),
                "residual_blocks": int(parameters.diffusion_residual_blocks),
                "timesteps": int(parameters.diffusion_timesteps),
            }
        )
        overrides["diffusion"] = diffusion
        return overrides

    @staticmethod
    def _row_parameters(row: pd.Series) -> NetworkCapacityParameterSet:
        return NetworkCapacityParameterSet(
            diffusion_hidden_dim=int(row["diffusion_hidden_dim"]),
            diffusion_residual_blocks=int(row["diffusion_residual_blocks"]),
            diffusion_timesteps=int(row["diffusion_timesteps"]),
        )

    def _write_plan(self) -> Path:
        experiment = load_config(self.config.experiment_config)
        project_count = sum(1 for _ in experiment.targets())
        family_count = len(experiment.data.families)
        sequential_unique = (
            len(self.config.grids.diffusion_hidden_dim)
            + len(self.config.grids.diffusion_residual_blocks)
            + len(self.config.grids.diffusion_timesteps)
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

    def run(self, *, dry_run: bool = False) -> NetworkCapacityParameterSet:
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
                "stage_01_diffusion_hidden_dim",
                stage_one_candidates(self.config),
                None,
            )
            eligible_one = stage_one[stage_one["safety_eligible"]].head(2)
            top_hidden_dims = tuple(
                int(value) for value in eligible_one["diffusion_hidden_dim"]
            )
            best_hidden_dim = top_hidden_dims[0]

            stage_two, _ = self._run_stage(
                "stage_02_diffusion_residual_blocks",
                stage_two_candidates(self.config, best_hidden_dim),
                baseline_metrics,
            )
            eligible_two = stage_two[stage_two["safety_eligible"]].head(2)
            top_residual_blocks = tuple(
                int(value) for value in eligible_two["diffusion_residual_blocks"]
            )
            best_residual_blocks = top_residual_blocks[0]

            stage_three, _ = self._run_stage(
                "stage_03_diffusion_timesteps",
                stage_three_candidates(
                    self.config,
                    best_hidden_dim,
                    best_residual_blocks,
                ),
                baseline_metrics,
            )
            eligible_three = stage_three[stage_three["safety_eligible"]].head(2)
            top_timesteps = tuple(
                int(value) for value in eligible_three["diffusion_timesteps"]
            )

            interactions = interaction_candidates(
                top_hidden_dims,
                top_residual_blocks,
                top_timesteps,
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
            comparison_png, comparison_pdf = create_network_capacity_figure(
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
                        "diffusion": {
                            "hidden_dim": winner.diffusion_hidden_dim,
                            "residual_blocks": winner.diffusion_residual_blocks,
                            "timesteps": winner.diffusion_timesteps,
                        }
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
            "Search IPGDO diffusion network-capacity parameters and run the "
            "10-seed winner CPDP experiment"
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
        config = load_network_capacity_search_config(Path(args.config))
        if args.check:
            print(
                "IPGDO_NETWORK_CAPACITY_SEARCH_OK "
                f"config={config.source_path} "
                f"screening_seeds={len(config.screening_seeds)} "
                f"confirmation_seeds={len(config.confirmation_seeds)} "
                f"final_cpdp_seeds={len(config.final_cpdp.seeds)}"
            )
            return 0
        engine = NetworkCapacitySearchEngine(config, workers=args.workers)
        engine.run(dry_run=bool(args.dry_run))
        return 0
    except KeyboardInterrupt:
        print("IPGDO parameter search interrupted; rerun to resume.", file=sys.stderr)
        return 130
    except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        print(f"IPGDO parameter search failed: {exc}", file=sys.stderr)
        return 1


__all__ = [
    "NetworkCapacityGrids",
    "NetworkCapacityParameterSet",
    "NetworkCapacitySearchConfig",
    "NetworkCapacitySearchEngine",
    "interaction_candidates",
    "load_network_capacity_search_config",
    "main",
    "stage_one_candidates",
    "stage_three_candidates",
    "stage_two_candidates",
]
