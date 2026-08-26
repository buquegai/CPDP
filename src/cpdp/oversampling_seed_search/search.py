from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

from cpdp.config import ExperimentConfig, load_config
from cpdp.data import load_all_projects, sources_for_target
from cpdp.metrics import METRIC_NAMES
from cpdp.oversampling_seed_search.cache import (
    mapping_signature,
    save_augmentation_cache,
    source_data_signature,
)
from cpdp.oversampling_validation.config import (
    ClassifierConfig,
    CPDPConfig,
    QualityConfig,
    SourceCVConfig,
    ValidationProfile,
)
from cpdp.randomness import seed_everything
from cpdp.types import LabeledProject


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "ipgdo_seed_search.yaml"


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


@dataclass(frozen=True)
class SeedSearchConfig:
    source_path: Path
    experiment_config: Path
    output_dir: Path
    method: str
    profile: str
    candidate_seeds: Tuple[int, ...]
    folds: int
    validation_fraction: float
    split_seed: int
    classifier_seed: int
    quality_seed: int
    metrics: Tuple[str, ...]
    classifier: ClassifierConfig
    quality: QualityConfig

    def validate(self) -> None:
        if not self.candidate_seeds:
            raise ValueError("search.candidate_seeds cannot be empty")
        if len(set(self.candidate_seeds)) != len(self.candidate_seeds):
            raise ValueError("search.candidate_seeds must be unique")
        if self.folds < 2:
            raise ValueError("source_cv.folds must be at least 2")
        if not 0.0 < self.validation_fraction < 0.5:
            raise ValueError("source_cv.validation_fraction must be in (0, 0.5)")
        unknown = sorted(set(self.metrics) - set(METRIC_NAMES))
        if unknown:
            raise ValueError(f"Unknown selection metrics: {', '.join(unknown)}")
        self.classifier.validate()
        self.quality.validate()


def load_seed_search_config(path: Path) -> SeedSearchConfig:
    source_path = path.resolve()
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping) or raw.get("schema_version") != 1:
        raise ValueError("Unsupported IPGDO seed-search schema_version")
    paths = _mapping(raw.get("paths"), "paths")
    search = _mapping(raw.get("search"), "search")
    source_cv = _mapping(raw.get("source_cv"), "source_cv")
    selection = _mapping(raw.get("selection"), "selection")
    configured_seeds = tuple(int(seed) for seed in search.get("candidate_seeds", ()))
    if configured_seeds:
        candidate_seeds = configured_seeds
    else:
        seed_start = int(search.get("candidate_seed_start", 2026072201))
        seed_count = int(search.get("candidate_seed_count", 10))
        if seed_count < 1:
            raise ValueError("search.candidate_seed_count must be positive")
        candidate_seeds = tuple(seed_start + offset for offset in range(seed_count))
    config = SeedSearchConfig(
        source_path=source_path,
        experiment_config=_expand_path(
            str(paths.get("experiment_config", "experiment.yaml")), source_path.parent
        ),
        output_dir=_expand_path(
            str(paths.get("output_dir", "../outputs/ipgdo_seed_search")),
            source_path.parent,
        ),
        method=str(search.get("method", "ipgdo_kan_jepa")),
        profile=str(search.get("profile", "full")),
        candidate_seeds=candidate_seeds,
        folds=int(source_cv.get("folds", 3)),
        validation_fraction=float(source_cv.get("validation_fraction", 0.2)),
        split_seed=int(source_cv.get("split_seed", 2026081201)),
        classifier_seed=int(source_cv.get("classifier_seed", 2026081202)),
        quality_seed=int(source_cv.get("quality_seed", 2026081203)),
        metrics=tuple(str(metric) for metric in selection.get("metrics", METRIC_NAMES)),
        classifier=ClassifierConfig(
            **_mapping(raw.get("classifier"), "classifier")
        ),
        quality=QualityConfig(**_mapping(raw.get("quality"), "quality")),
    )
    config.validate()
    return config


def _profile(config: SeedSearchConfig) -> ValidationProfile:
    return ValidationProfile(
        name="ipgdo_seed_search",
        oversampling_profile=config.profile,
        seeds=config.candidate_seeds,
        variants=("full",),
        source_cv=SourceCVConfig(
            enabled=True,
            folds=config.folds,
            validation_fraction=config.validation_fraction,
        ),
        cpdp=CPDPConfig(enabled=False, validation_fraction=config.validation_fraction),
        classifier=config.classifier,
        quality=config.quality,
    )


def _safe_target_component(target_key: str) -> str:
    return target_key.replace("/", "__").replace("\\", "__")


def _fold_path(root: Path, target_key: str, seed: int, fold: int) -> Path:
    return (
        root
        / "runs"
        / _safe_target_component(target_key)
        / f"seed_{seed}"
        / f"fold_{fold}.json"
    )


def _successful_payload(path: Path, search_signature: str) -> Optional[Mapping[str, Any]]:
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


def _fold_payload(
    *,
    target_key: str,
    candidate_seed: int,
    fold: int,
    result: Any,
    search_signature: str,
    duration_seconds: float,
) -> Mapping[str, Any]:
    return {
        "schema_version": 1,
        "status": "success",
        "search_signature": search_signature,
        "target": target_key,
        "candidate_seed": int(candidate_seed),
        "fold": int(fold),
        "selection_protocol": "source-only-cv",
        "target_labels_used": False,
        "utility": result.utility,
        "quality": result.quality,
        "duration_seconds": float(duration_seconds),
        "finished_at": _utc_now(),
    }


def _project_metric_rows(payload: Mapping[str, Any]) -> Iterable[Dict[str, Any]]:
    per_project = payload["utility"]["augmented"]["per_project_metrics"]
    for project, metrics in per_project.items():
        yield {
            "target": payload["target"],
            "candidate_seed": int(payload["candidate_seed"]),
            "fold": int(payload["fold"]),
            "source_project": str(project),
            **{metric: metrics.get(metric) for metric in METRIC_NAMES},
        }


def _quality_rows(payload: Mapping[str, Any]) -> Iterable[Dict[str, Any]]:
    for record in payload.get("quality", ()):
        yield {
            "target": payload["target"],
            "candidate_seed": int(payload["candidate_seed"]),
            "fold": int(payload["fold"]),
            "source_project": str(record["project"]),
            **dict(record.get("metrics", {})),
        }


_QUALITY_DIRECTIONS = {
    "normalized_wasserstein_mean": True,
    "rbf_mmd": True,
    "correlation_frobenius": True,
    "exact_duplicate_rate": True,
    "near_duplicate_rate": True,
    "defect_manifold_in_range_rate": False,
    "heldout_defect_coverage": False,
    "defect_neighborhood_purity": False,
}


def rank_target_seeds(
    metric_rows: pd.DataFrame,
    quality_rows: pd.DataFrame,
    metrics: Sequence[str],
) -> pd.DataFrame:
    grouped = (
        metric_rows.groupby(["candidate_seed", "source_project"], as_index=False)[
            list(metrics)
        ]
        .mean(numeric_only=True)
    )
    seeds = tuple(sorted(int(seed) for seed in grouped["candidate_seed"].unique()))
    utility_ranks: Dict[int, list[float]] = {seed: [] for seed in seeds}
    for _, project_rows in grouped.groupby("source_project", sort=False):
        indexed = project_rows.set_index("candidate_seed")
        for metric in metrics:
            values = pd.to_numeric(indexed[metric], errors="coerce")
            finite = values[np.isfinite(values)]
            ranked = finite.rank(method="average", ascending=False)
            worst = float(len(seeds) + 1)
            for seed in seeds:
                utility_ranks[seed].append(float(ranked.get(seed, worst)))

    quality_macro = quality_rows.groupby("candidate_seed", as_index=True).mean(
        numeric_only=True
    )
    quality_ranks: Dict[int, list[float]] = {seed: [] for seed in seeds}
    for metric, ascending in _QUALITY_DIRECTIONS.items():
        if metric not in quality_macro:
            continue
        values = pd.to_numeric(quality_macro[metric], errors="coerce")
        finite = values[np.isfinite(values)]
        if finite.empty:
            continue
        ranked = finite.rank(method="average", ascending=ascending)
        worst = float(len(seeds) + 1)
        for seed in seeds:
            quality_ranks[seed].append(float(ranked.get(seed, worst)))

    candidate_macro = grouped.groupby("candidate_seed", as_index=True).mean(
        numeric_only=True
    )
    rows = []
    for seed in seeds:
        row: Dict[str, Any] = {
            "candidate_seed": seed,
            "mean_utility_rank": float(np.mean(utility_ranks[seed])),
            "utility_rank_cells": len(utility_ranks[seed]),
            "mean_quality_rank": (
                float(np.mean(quality_ranks[seed]))
                if quality_ranks[seed]
                else float("inf")
            ),
            "quality_rank_cells": len(quality_ranks[seed]),
        }
        for metric in metrics:
            row[f"macro_{metric}"] = float(candidate_macro.loc[seed, metric])
        if seed in quality_macro.index:
            for metric in _QUALITY_DIRECTIONS:
                if metric in quality_macro:
                    row[f"quality_{metric}"] = float(quality_macro.loc[seed, metric])
        rows.append(row)
    ranking = pd.DataFrame(rows)
    tie_break_metrics = [
        column
        for column in ("macro_mcc", "macro_gmean", "macro_auc", "macro_auprc")
        if column in ranking.columns
    ]
    ranking = ranking.sort_values(
        ["mean_utility_rank", "mean_quality_rank", *tie_break_metrics, "candidate_seed"],
        ascending=[True, True, *([False] * len(tie_break_metrics)), True],
        na_position="last",
    ).reset_index(drop=True)
    ranking.insert(0, "selection_position", np.arange(1, len(ranking) + 1))
    return ranking


class IPGDOSeedSearch:
    def __init__(self, config: SeedSearchConfig) -> None:
        self.config = config
        self.experiment: ExperimentConfig = load_config(config.experiment_config)
        params = self.experiment.method_params(config.profile, config.method)
        oversampling = params.get("oversampling")
        if not isinstance(oversampling, Mapping):
            raise TypeError(f"{config.method}.oversampling must be a mapping")
        self.oversampling_config = dict(oversampling)
        self.projects: Optional[Mapping[str, LabeledProject]] = None
        self.profile = _profile(config)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def _target_signature(
        self, target_key: str, sources: Sequence[LabeledProject]
    ) -> str:
        return mapping_signature(
            {
                "schema_version": 1,
                "target": target_key,
                "candidate_seeds": self.config.candidate_seeds,
                "folds": self.config.folds,
                "validation_fraction": self.config.validation_fraction,
                "split_seed": self.config.split_seed,
                "classifier_seed": self.config.classifier_seed,
                "quality_seed": self.config.quality_seed,
                "metrics": self.config.metrics,
                "classifier": asdict(self.config.classifier),
                "quality": asdict(self.config.quality),
                "oversampling_config": self.oversampling_config,
                "source_data_signature": source_data_signature(sources),
            }
        )

    def _run_target(
        self,
        family: str,
        target_name: str,
        *,
        force: bool,
    ) -> Mapping[str, Any]:
        target_key = f"{family}/{target_name}"
        if self.projects is None:
            raise RuntimeError("Dataset projects have not been loaded")
        sources = tuple(sources_for_target(self.projects, family, target_name))
        if not sources:
            raise ValueError(f"No source projects are available for {target_key}")
        signature = self._target_signature(target_key, sources)
        payloads = []
        for candidate_seed in self.config.candidate_seeds:
            for fold in range(self.config.folds):
                path = _fold_path(
                    self.config.output_dir, target_key, candidate_seed, fold
                )
                payload = None if force else _successful_payload(path, signature)
                if payload is None:
                    from cpdp.oversampling_validation.protocols import (
                        run_source_cv_fold,
                    )
                    from cpdp.oversampling_validation.strategies import (
                        CurrentPipelineStrategy,
                    )

                    print(
                        f"SEED_SEARCH_RUN target={target_key} "
                        f"candidate={candidate_seed} fold={fold + 1}/{self.config.folds}",
                        flush=True,
                    )
                    started = time.perf_counter()
                    seed_everything(candidate_seed)
                    result = run_source_cv_fold(
                        sources,
                        CurrentPipelineStrategy(self.oversampling_config),
                        self.profile,
                        candidate_seed,
                        fold,
                        split_seed=self.config.split_seed,
                        oversampling_seed=candidate_seed,
                        classifier_seed=self.config.classifier_seed,
                        quality_seed=self.config.quality_seed,
                    )
                    payload = _fold_payload(
                        target_key=target_key,
                        candidate_seed=candidate_seed,
                        fold=fold,
                        result=result,
                        search_signature=signature,
                        duration_seconds=time.perf_counter() - started,
                    )
                    _atomic_json(path, payload)
                else:
                    print(
                        f"SEED_SEARCH_SKIP target={target_key} "
                        f"candidate={candidate_seed} fold={fold + 1}/{self.config.folds}",
                        flush=True,
                    )
                payloads.append(payload)

        metric_rows = pd.DataFrame(
            [row for payload in payloads for row in _project_metric_rows(payload)]
        )
        quality_rows = pd.DataFrame(
            [row for payload in payloads for row in _quality_rows(payload)]
        )
        target_dir = self.config.output_dir / "summary" / _safe_target_component(target_key)
        ranking = rank_target_seeds(metric_rows, quality_rows, self.config.metrics)
        _atomic_csv(target_dir / "source_cv_metrics.csv", metric_rows)
        _atomic_csv(target_dir / "generation_quality.csv", quality_rows)
        _atomic_csv(target_dir / "seed_rankings.csv", ranking)
        best_seed = int(ranking.iloc[0]["candidate_seed"])

        cache_path = (
            self.config.output_dir
            / "cache"
            / _safe_target_component(target_key)
            / f"seed_{best_seed}.npz"
        )
        try:
            from cpdp.oversampling_seed_search.cache import load_augmentation_cache

            load_augmentation_cache(
                cache_path,
                target_key=target_key,
                oversampling_seed=best_seed,
                oversampling_config=self.oversampling_config,
                sources=sources,
            )
            print(f"AUGMENTATION_CACHE_REUSE target={target_key} seed={best_seed}", flush=True)
        except (FileNotFoundError, KeyError, ValueError):
            from cpdp.oversampling.pipeline import IPGDOOversampler

            print(f"AUGMENTATION_CACHE_BUILD target={target_key} seed={best_seed}", flush=True)
            seed_everything(best_seed)
            augmentation = IPGDOOversampler.from_mapping(
                self.oversampling_config
            ).fit_resample(sources, best_seed)
            save_augmentation_cache(
                cache_path,
                augmentation,
                target_key=target_key,
                oversampling_seed=best_seed,
                oversampling_config=self.oversampling_config,
                sources=sources,
            )

        winner = ranking.iloc[0].to_dict()
        return {
            "best_seed": best_seed,
            "cache_file": os.path.relpath(cache_path, self.config.output_dir),
            "ranking_file": os.path.relpath(
                target_dir / "seed_rankings.csv", self.config.output_dir
            ),
            "source_projects": [source.key for source in sources],
            "search_signature": signature,
            "selection": {
                key: (value.item() if isinstance(value, np.generic) else value)
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
            if len(selected) != len(requested):
                known = {f"{family}/{target}".lower() for family, target in configured}
                unknown = sorted(requested - known)
                raise KeyError(f"Unknown target(s): {', '.join(unknown)}")
        else:
            selected = configured
        plan = {
            "targets": len(selected),
            "candidate_seeds": len(self.config.candidate_seeds),
            "folds": self.config.folds,
            "ipgdo_training_runs": len(selected)
            * (len(self.config.candidate_seeds) * self.config.folds + 1),
            "selection_protocol": "source-only-cv",
            "target_labels_used": False,
        }
        _atomic_yaml(self.config.output_dir / "search_plan.yaml", plan)
        if dry_run:
            print(f"SEED_SEARCH_DRY_RUN {plan}", flush=True)
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
                f"SEED_SEARCH_TARGET {index}/{len(selected)} target={target_key}",
                flush=True,
            )
            existing[target_key] = self._run_target(
                family, target_name, force=force
            )
            _atomic_yaml(
                map_path,
                {
                    "schema_version": 1,
                    "selection_protocol": "source-only-cv",
                    "target_labels_used": False,
                    "method": self.config.method,
                    "profile": self.config.profile,
                    "candidate_seeds": list(self.config.candidate_seeds),
                    "fixed_randomness": {
                        "split_seed": self.config.split_seed,
                        "classifier_seed": self.config.classifier_seed,
                        "quality_seed": self.config.quality_seed,
                    },
                    "updated_at": _utc_now(),
                    "targets": existing,
                },
            )
        print(f"SEED_SEARCH_COMPLETE seed_map={map_path}", flush=True)
        return map_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select one leakage-free IPGDO seed for each CPDP target"
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--target", action="append", help="family/project; repeatable")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    search = IPGDOSeedSearch(load_seed_search_config(Path(args.config)))
    search.run(targets=args.target, force=args.force, dry_run=args.dry_run)
    return 0


__all__ = [
    "IPGDOSeedSearch",
    "SeedSearchConfig",
    "load_seed_search_config",
    "main",
    "rank_target_seeds",
]
