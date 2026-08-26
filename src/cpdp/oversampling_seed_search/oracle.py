from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

from cpdp.config import ExperimentConfig, load_config
from cpdp.data import load_all_projects, sources_for_target
from cpdp.metrics import METRIC_NAMES, evaluate_predictions
from cpdp.oversampling.pipeline import IPGDOOversampler
from cpdp.oversampling.types import AugmentationResult
from cpdp.oversampling_seed_search.cache import (
    load_augmentation_cache,
    mapping_signature,
    save_augmentation_cache,
    source_data_signature,
)
from cpdp.predictors.kan_jepa import KANJEPAPredictor
from cpdp.randomness import seed_everything
from cpdp.types import LabeledProject, PredictionResult


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "ipgdo_oracle_seed_search.yaml"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _expand_path(value: str, base: Path) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(value)))
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            _jsonable(payload), handle, ensure_ascii=False, indent=2, sort_keys=True
        )
    os.replace(temporary, path)


def _atomic_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(
            _jsonable(payload), handle, allow_unicode=True, sort_keys=False
        )
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def _safe_component(value: str) -> str:
    return value.replace("/", "__").replace("\\", "__")


@dataclass(frozen=True)
class OracleSeedSearchConfig:
    source_path: Path
    experiment_config: Path
    output_dir: Path
    method: str
    profile: str
    candidate_seeds: Tuple[int, ...]
    predictor_seed: int
    metrics: Tuple[str, ...]
    retain_candidate_caches: bool

    def validate(self) -> None:
        if self.method != "ipgdo_kan_jepa":
            raise ValueError("Oracle search must use ipgdo_kan_jepa")
        if not self.candidate_seeds:
            raise ValueError("search.candidate_seeds cannot be empty")
        if len(set(self.candidate_seeds)) != len(self.candidate_seeds):
            raise ValueError("search.candidate_seeds must be unique")
        unknown = sorted(set(self.metrics) - set(METRIC_NAMES))
        if unknown:
            raise ValueError(f"Unknown selection metrics: {', '.join(unknown)}")


def load_oracle_seed_search_config(path: Path) -> OracleSeedSearchConfig:
    source_path = path.resolve()
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping) or raw.get("schema_version") != 1:
        raise ValueError("Unsupported oracle seed-search schema_version")
    paths = _mapping(raw.get("paths"), "paths")
    search = _mapping(raw.get("search"), "search")
    selection = _mapping(raw.get("selection"), "selection")
    storage = _mapping(raw.get("storage"), "storage")
    explicit_seeds = tuple(int(seed) for seed in search.get("candidate_seeds", ()))
    if explicit_seeds:
        candidate_seeds = explicit_seeds
    else:
        seed_start = int(search.get("candidate_seed_start", 2026072201))
        seed_count = int(search.get("candidate_seed_count", 100))
        if seed_count < 1:
            raise ValueError("search.candidate_seed_count must be positive")
        candidate_seeds = tuple(seed_start + offset for offset in range(seed_count))
    config = OracleSeedSearchConfig(
        source_path=source_path,
        experiment_config=_expand_path(
            str(paths.get("experiment_config", "experiment.yaml")),
            source_path.parent,
        ),
        output_dir=_expand_path(
            str(paths.get("output_dir", "../outputs/ipgdo_oracle_seed_search")),
            source_path.parent,
        ),
        method=str(search.get("method", "ipgdo_kan_jepa")),
        profile=str(search.get("profile", "full")),
        candidate_seeds=candidate_seeds,
        predictor_seed=int(search.get("predictor_seed", 2026081601)),
        metrics=tuple(
            str(metric) for metric in selection.get("metrics", METRIC_NAMES)
        ),
        retain_candidate_caches=bool(
            storage.get("retain_candidate_caches", False)
        ),
    )
    config.validate()
    return config


def rank_oracle_candidates(
    metric_rows: pd.DataFrame, metrics: Sequence[str]
) -> pd.DataFrame:
    required = {"candidate_seed", *metrics}
    missing = sorted(required - set(metric_rows.columns))
    if missing:
        raise ValueError(f"Missing oracle ranking columns: {', '.join(missing)}")
    ranking = metric_rows.copy()
    rank_columns = []
    for metric in metrics:
        rank_column = f"rank_{metric}"
        values = pd.to_numeric(ranking[metric], errors="coerce")
        finite = values[np.isfinite(values)]
        metric_ranks = finite.rank(method="average", ascending=False)
        ranking[rank_column] = metric_ranks.reindex(ranking.index).fillna(
            len(ranking) + 1
        )
        rank_columns.append(rank_column)
    ranking["mean_metric_rank"] = ranking[rank_columns].mean(axis=1)
    tie_breakers = [
        metric
        for metric in ("mcc", "gmean", "auc", "auprc", "f1", "popt20")
        if metric in metrics
    ]
    ranking = ranking.sort_values(
        ["mean_metric_rank", *tie_breakers, "candidate_seed"],
        ascending=[True, *([False] * len(tie_breakers)), True],
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)
    ranking.insert(0, "selection_position", np.arange(1, len(ranking) + 1))
    return ranking


def _successful_payload(
    path: Path, search_signature: str
) -> Optional[Mapping[str, Any]]:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("status") != "success":
        return None
    if payload.get("search_signature") != search_signature:
        return None
    return payload


def _evaluation_arrays(
    target: LabeledProject, prediction: PredictionResult
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores = np.asarray(prediction.scores, dtype=np.float64).reshape(-1)
    if prediction.evaluated_row_ids is None:
        indices = np.arange(target.features.n_samples, dtype=np.int64)
    else:
        requested = np.asarray(prediction.evaluated_row_ids).reshape(-1)
        positions = {
            int(row_id): index
            for index, row_id in enumerate(target.features.row_ids.tolist())
        }
        if len(set(int(row_id) for row_id in requested)) != len(requested):
            raise ValueError("evaluated_row_ids contains duplicates")
        try:
            indices = np.asarray(
                [positions[int(row_id)] for row_id in requested], dtype=np.int64
            )
        except KeyError as exc:
            raise ValueError("Prediction references an unknown target row") from exc
    if len(scores) != len(indices):
        raise ValueError("Prediction score count does not match target rows")
    return (
        np.asarray(target.labels[indices], dtype=np.int64),
        scores,
        np.asarray(target.features.effort[indices], dtype=np.float64),
    )


class IPGDOOracleSeedSearch:
    """Target-label oracle search for diagnostic use only.

    The real target labels are used to rank oversampling seeds. Results from
    this search are not an unbiased estimate of CPDP generalization.
    """

    def __init__(self, config: OracleSeedSearchConfig) -> None:
        self.config = config
        self.experiment: ExperimentConfig = load_config(config.experiment_config)
        params = self.experiment.method_params(config.profile, config.method)
        oversampling = params.get("oversampling")
        predictor = params.get("predictor")
        if not isinstance(oversampling, Mapping):
            raise TypeError(f"{config.method}.oversampling must be a mapping")
        if not isinstance(predictor, Mapping):
            raise TypeError(f"{config.method}.predictor must be a mapping")
        self.oversampling_config = dict(oversampling)
        self.predictor_config = dict(predictor)
        self.projects: Optional[Mapping[str, LabeledProject]] = None
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def _target_signature(
        self,
        target: LabeledProject,
        sources: Sequence[LabeledProject],
    ) -> str:
        return mapping_signature(
            {
                "schema_version": 1,
                "selection_protocol": "target-label-oracle-full-cpdp",
                "target": target.key,
                "target_data_signature": source_data_signature((target,)),
                "source_data_signature": source_data_signature(sources),
                "candidate_seeds": self.config.candidate_seeds,
                "predictor_seed": self.config.predictor_seed,
                "metrics": self.config.metrics,
                "oversampling_config": self.oversampling_config,
                "predictor_config": self.predictor_config,
            }
        )

    def _run_path(self, target_key: str, candidate_seed: int) -> Path:
        return (
            self.config.output_dir
            / "runs"
            / _safe_component(target_key)
            / f"seed_{candidate_seed}.json"
        )

    def _cache_path(self, target_key: str, candidate_seed: int) -> Path:
        return (
            self.config.output_dir
            / "cache"
            / _safe_component(target_key)
            / f"seed_{candidate_seed}.npz"
        )

    def _load_or_build_augmentation(
        self,
        target_key: str,
        candidate_seed: int,
        sources: Sequence[LabeledProject],
    ) -> AugmentationResult:
        cache_path = self._cache_path(target_key, candidate_seed)
        try:
            augmentation = load_augmentation_cache(
                cache_path,
                target_key=target_key,
                oversampling_seed=candidate_seed,
                oversampling_config=self.oversampling_config,
                sources=sources,
            )
            print(
                f"ORACLE_CACHE_REUSE target={target_key} seed={candidate_seed}",
                flush=True,
            )
            return augmentation
        except (FileNotFoundError, KeyError, ValueError):
            print(
                f"ORACLE_CACHE_BUILD target={target_key} seed={candidate_seed}",
                flush=True,
            )
            seed_everything(candidate_seed)
            augmentation = IPGDOOversampler.from_mapping(
                self.oversampling_config
            ).fit_resample(sources, candidate_seed)
            save_augmentation_cache(
                cache_path,
                augmentation,
                target_key=target_key,
                oversampling_seed=candidate_seed,
                oversampling_config=self.oversampling_config,
                sources=sources,
            )
            return augmentation

    def _run_candidate(
        self,
        *,
        target: LabeledProject,
        sources: Sequence[LabeledProject],
        candidate_seed: int,
        search_signature: str,
        force: bool,
    ) -> Mapping[str, Any]:
        path = self._run_path(target.key, candidate_seed)
        payload = None if force else _successful_payload(path, search_signature)
        if payload is not None:
            print(
                f"ORACLE_SKIP target={target.key} seed={candidate_seed}", flush=True
            )
            return payload

        print(
            f"ORACLE_RUN target={target.key} oversampling_seed={candidate_seed} "
            f"predictor_seed={self.config.predictor_seed}",
            flush=True,
        )
        started = time.perf_counter()
        augmentation = self._load_or_build_augmentation(
            target.key, candidate_seed, sources
        )
        seed_everything(self.config.predictor_seed)
        predictor = KANJEPAPredictor.from_mapping(self.predictor_config)
        prediction = predictor.fit_predict(
            augmentation.projects,
            target.without_labels(),
            self.config.predictor_seed,
        )
        labels, scores, effort = _evaluation_arrays(target, prediction)
        metrics = evaluate_predictions(
            labels, scores, effort, float(prediction.threshold)
        )
        payload = {
            "schema_version": 1,
            "status": "success",
            "search_signature": search_signature,
            "selection_protocol": "target-label-oracle-full-cpdp",
            "target": target.key,
            "target_labels_used_for_evaluation": True,
            "target_labels_used_for_seed_selection": True,
            "valid_for_unbiased_formal_comparison": False,
            "candidate_seed": int(candidate_seed),
            "predictor_seed": int(self.config.predictor_seed),
            "threshold": float(prediction.threshold),
            "metrics": {name: metrics[name] for name in self.config.metrics},
            "duration_seconds": float(time.perf_counter() - started),
            "source_projects": [source.key for source in sources],
            "predictor_audit": {
                "data_contract": prediction.metadata.get("data_contract", {}),
                "model_selection": prediction.metadata.get("model_selection", {}),
                "final_training": prediction.metadata.get("final_training", {}),
            },
            "finished_at": _utc_now(),
        }
        _atomic_json(path, payload)
        metric_text = " ".join(
            f"{metric}={metrics[metric]:.4f}" for metric in self.config.metrics
        )
        print(
            f"ORACLE_OK target={target.key} oversampling_seed={candidate_seed} "
            f"{metric_text}",
            flush=True,
        )
        return payload

    def _cleanup_loser_caches(
        self, target_key: str, best_seed: int
    ) -> None:
        if self.config.retain_candidate_caches:
            return
        for candidate_seed in self.config.candidate_seeds:
            if candidate_seed == best_seed:
                continue
            cache_path = self._cache_path(target_key, candidate_seed)
            for path in (cache_path, cache_path.with_suffix(".json")):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass

    def _run_target(
        self, family: str, target_name: str, *, force: bool
    ) -> Mapping[str, Any]:
        if self.projects is None:
            raise RuntimeError("Dataset projects have not been loaded")
        target = self.projects[f"{family}/{target_name}"]
        sources = tuple(sources_for_target(self.projects, family, target_name))
        signature = self._target_signature(target, sources)
        payloads = [
            self._run_candidate(
                target=target,
                sources=sources,
                candidate_seed=candidate_seed,
                search_signature=signature,
                force=force,
            )
            for candidate_seed in self.config.candidate_seeds
        ]
        metric_rows = pd.DataFrame(
            [
                {
                    "target": payload["target"],
                    "candidate_seed": int(payload["candidate_seed"]),
                    "predictor_seed": int(payload["predictor_seed"]),
                    "duration_seconds": float(payload["duration_seconds"]),
                    **dict(payload["metrics"]),
                }
                for payload in payloads
            ]
        )
        ranking = rank_oracle_candidates(metric_rows, self.config.metrics)
        best_seed = int(ranking.iloc[0]["candidate_seed"])
        target_dir = (
            self.config.output_dir / "summary" / _safe_component(target.key)
        )
        metrics_path = target_dir / "candidate_metrics.csv"
        ranking_path = target_dir / "candidate_rankings.csv"
        _atomic_csv(metrics_path, metric_rows)
        _atomic_csv(ranking_path, ranking)
        winner_cache = self._cache_path(target.key, best_seed)
        load_augmentation_cache(
            winner_cache,
            target_key=target.key,
            oversampling_seed=best_seed,
            oversampling_config=self.oversampling_config,
            sources=sources,
        )
        self._cleanup_loser_caches(target.key, best_seed)
        winner = ranking.iloc[0].to_dict()
        return {
            "best_seed": best_seed,
            "cache_file": os.path.relpath(winner_cache, self.config.output_dir),
            "ranking_file": os.path.relpath(ranking_path, self.config.output_dir),
            "metrics_file": os.path.relpath(metrics_path, self.config.output_dir),
            "predictor_seed": int(self.config.predictor_seed),
            "source_projects": [source.key for source in sources],
            "search_signature": signature,
            "selection": {
                key: value.item() if isinstance(value, np.generic) else value
                for key, value in winner.items()
            },
        }

    def run(
        self,
        *,
        targets: Optional[Sequence[str]] = None,
        force: bool = False,
        dry_run: bool = False,
    ) -> Path:
        configured = tuple(self.experiment.targets())
        if targets:
            requested = {value.replace("\\", "/").lower() for value in targets}
            selected = tuple(
                (family, target)
                for family, target in configured
                if f"{family}/{target}".lower() in requested
            )
            known = {f"{family}/{target}".lower() for family, target in configured}
            unknown = sorted(requested - known)
            if unknown:
                raise KeyError(f"Unknown target(s): {', '.join(unknown)}")
        else:
            selected = configured
        task_count = len(selected) * len(self.config.candidate_seeds)
        plan = {
            "targets": len(selected),
            "candidate_seeds_per_target": len(self.config.candidate_seeds),
            "predictor_seed": int(self.config.predictor_seed),
            "ipgdo_runs": task_count,
            "kan_jepa_runs": task_count,
            "selection_protocol": "target-label-oracle-full-cpdp",
            "target_labels_used_for_seed_selection": True,
            "valid_for_unbiased_formal_comparison": False,
        }
        _atomic_yaml(self.config.output_dir / "search_plan.yaml", plan)
        if dry_run:
            print(f"ORACLE_SEARCH_DRY_RUN {plan}", flush=True)
            return self.config.output_dir / "best_oversampling_seeds.yaml"

        self.projects = load_all_projects(self.experiment.data)
        map_path = self.config.output_dir / "best_oversampling_seeds.yaml"
        existing: Dict[str, Any] = {}
        if map_path.is_file() and not force:
            with map_path.open("r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle)
            if isinstance(loaded, Mapping):
                existing = dict(_mapping(loaded.get("targets"), "targets"))
        for index, (family, target_name) in enumerate(selected, start=1):
            target_key = f"{family}/{target_name}"
            print(
                f"ORACLE_TARGET {index}/{len(selected)} target={target_key}",
                flush=True,
            )
            existing[target_key] = self._run_target(
                family, target_name, force=force
            )
            _atomic_yaml(
                map_path,
                {
                    "schema_version": 1,
                    "selection_protocol": "target-label-oracle-full-cpdp",
                    "target_labels_used_for_seed_selection": True,
                    "valid_for_unbiased_formal_comparison": False,
                    "method": self.config.method,
                    "profile": self.config.profile,
                    "predictor_seed": int(self.config.predictor_seed),
                    "candidate_seeds": list(self.config.candidate_seeds),
                    "updated_at": _utc_now(),
                    "targets": existing,
                },
            )
        print(f"ORACLE_SEARCH_COMPLETE seed_map={map_path}", flush=True)
        return map_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run full CPDP for 100 IPGDO seeds with one fixed KAN-JEPA seed. "
            "This oracle search uses target labels for seed selection."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--target", action="append", help="family/project; repeatable")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    search = IPGDOOracleSeedSearch(
        load_oracle_seed_search_config(Path(args.config))
    )
    search.run(targets=args.target, force=args.force, dry_run=args.dry_run)
    return 0


__all__ = [
    "IPGDOOracleSeedSearch",
    "OracleSeedSearchConfig",
    "load_oracle_seed_search_config",
    "main",
    "rank_oracle_candidates",
]
