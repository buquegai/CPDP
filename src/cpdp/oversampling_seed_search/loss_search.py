from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import time
import traceback
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml

from cpdp.config import ExperimentConfig, load_config
from cpdp.data import load_all_projects, sources_for_target
from cpdp.oversampling.pipeline import IPGDOOversampler
from cpdp.oversampling_seed_search.cache import (
    load_augmentation_cache,
    mapping_signature,
    save_augmentation_cache,
    source_data_signature,
)
from cpdp.randomness import seed_everything


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "ipgdo_loss_seed_search.yaml"

LOSS_FIELDS = (
    "total_loss",
    "noise_loss",
    "physics_loss",
    "consistency_loss",
    "semantic_loss",
)
LOSS_RECORD_FIELDS = (
    "epoch",
    "batch_index",
    "global_batch",
    *LOSS_FIELDS,
    "physics_scale",
    "learning_rate",
)

_WORKER_PROJECTS: Optional[Mapping[str, Any]] = None
_WORKER_OVERSAMPLER: Optional[IPGDOOversampler] = None
_WORKER_TAIL_BATCH_COUNT = 10
_WORKER_SCORE_FIELD = "total_loss"
_WORKER_SEARCH_SIGNATURE = ""


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


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True),
    )


def _atomic_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(
        path,
        yaml.safe_dump(
            _jsonable(payload), allow_unicode=True, sort_keys=False
        ),
    )


def _atomic_csv(
    path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _safe_component(value: str) -> str:
    return value.replace("/", "__").replace("\\", "__").replace(" ", "_")


def _run_path(output_dir: Path, family: str, target: str, seed: int) -> Path:
    return output_dir / "runs" / family / target / f"seed_{seed}.json"


def _ranking_path(output_dir: Path, target_key: str) -> Path:
    return (
        output_dir
        / "summary"
        / _safe_component(target_key)
        / "candidate_loss_rankings.csv"
    )


@dataclass(frozen=True)
class LossSeedSearchConfig:
    source_path: Path
    experiment_config: Path
    output_dir: Path
    method: str
    profile: str
    candidate_seeds: Tuple[int, ...]
    final_batch_count: int
    score_field: str
    gpu_workers: int
    build_winner_cache: bool

    def validate(self) -> None:
        if not self.candidate_seeds:
            raise ValueError("search.candidate_seeds cannot be empty")
        if len(set(self.candidate_seeds)) != len(self.candidate_seeds):
            raise ValueError("search.candidate_seeds must be unique")
        if self.final_batch_count < 1:
            raise ValueError("selection.final_batch_count must be positive")
        if self.score_field not in LOSS_FIELDS:
            raise ValueError(
                f"selection.score_field must be one of: {', '.join(LOSS_FIELDS)}"
            )
        if self.gpu_workers < 1:
            raise ValueError("runtime.gpu_workers must be positive")


def load_loss_seed_search_config(path: Path) -> LossSeedSearchConfig:
    source_path = path.resolve()
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping) or raw.get("schema_version") != 1:
        raise ValueError("Unsupported IPGDO loss-seed-search schema_version")
    paths = _mapping(raw.get("paths"), "paths")
    search = _mapping(raw.get("search"), "search")
    selection = _mapping(raw.get("selection"), "selection")
    runtime = _mapping(raw.get("runtime"), "runtime")
    winner_cache = _mapping(raw.get("winner_cache"), "winner_cache")
    explicit_seeds = tuple(int(seed) for seed in search.get("candidate_seeds", ()))
    if explicit_seeds:
        candidate_seeds = explicit_seeds
    else:
        start = int(search.get("candidate_seed_start", 2026072201))
        count = int(search.get("candidate_seed_count", 100))
        if count < 1:
            raise ValueError("search.candidate_seed_count must be positive")
        candidate_seeds = tuple(start + offset for offset in range(count))
    config = LossSeedSearchConfig(
        source_path=source_path,
        experiment_config=_expand_path(
            str(paths.get("experiment_config", "experiment.yaml")),
            source_path.parent,
        ),
        output_dir=_expand_path(
            str(paths.get("output_dir", "../outputs/ipgdo_loss_seed_search")),
            source_path.parent,
        ),
        method=str(search.get("method", "ipgdo_kan_jepa")),
        profile=str(search.get("profile", "full")),
        candidate_seeds=candidate_seeds,
        final_batch_count=int(selection.get("final_batch_count", 10)),
        score_field=str(selection.get("score_field", "total_loss")),
        gpu_workers=int(runtime.get("gpu_workers", 2)),
        build_winner_cache=bool(winner_cache.get("enabled", True)),
    )
    config.validate()
    return config


def _selected_targets(
    experiment: ExperimentConfig, requested: Sequence[str]
) -> Tuple[Tuple[str, str], ...]:
    configured = tuple(experiment.targets())
    if not requested:
        return configured
    lookup = {
        f"{family}/{target}".lower(): (family, target)
        for family, target in configured
    }
    selected = []
    for value in requested:
        key = value.replace("\\", "/").strip().lower()
        if key not in lookup:
            raise KeyError(f"Unknown target: {value}")
        selected.append(lookup[key])
    return tuple(dict.fromkeys(selected))


def _search_signature(
    config: LossSeedSearchConfig,
    oversampling_config: Mapping[str, Any],
) -> str:
    experiment_digest = hashlib.sha256(
        config.experiment_config.read_bytes()
    ).hexdigest()
    return mapping_signature(
        {
            "schema_version": 1,
            "selection_protocol": "source-training-final-batch-loss",
            "experiment_config_sha256": experiment_digest,
            "method": config.method,
            "profile": config.profile,
            "candidate_seeds": config.candidate_seeds,
            "final_batch_count": config.final_batch_count,
            "score_field": config.score_field,
            "oversampling": oversampling_config,
        }
    )


def _successful_payload(
    path: Path,
    *,
    search_signature: str,
    source_signature: str,
) -> Optional[Mapping[str, Any]]:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("status") != "success":
        return None
    if payload.get("search_signature") != search_signature:
        return None
    if payload.get("source_data_signature") != source_signature:
        return None
    return payload


def _worker_init(
    config_path: str,
    profile: str,
    method: str,
    final_batch_count: int,
    score_field: str,
    search_signature: str,
) -> None:
    global _WORKER_PROJECTS
    global _WORKER_OVERSAMPLER
    global _WORKER_TAIL_BATCH_COUNT
    global _WORKER_SCORE_FIELD
    global _WORKER_SEARCH_SIGNATURE
    experiment = load_config(Path(config_path))
    _WORKER_PROJECTS = load_all_projects(experiment.data)
    method_params = experiment.method_params(profile, method)
    oversampling = method_params.get("oversampling")
    if not isinstance(oversampling, Mapping):
        raise ValueError(f"{profile}/{method} has no oversampling configuration")
    _WORKER_OVERSAMPLER = IPGDOOversampler.from_mapping(oversampling)
    _WORKER_TAIL_BATCH_COUNT = final_batch_count
    _WORKER_SCORE_FIELD = score_field
    _WORKER_SEARCH_SIGNATURE = search_signature


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


def _run_one(
    task: Tuple[str, str, int, str, str, bool]
) -> Dict[str, Any]:
    family, target, candidate_seed, output_value, source_signature, force = task
    output_dir = Path(output_value)
    path = _run_path(output_dir, family, target, candidate_seed)
    target_key = f"{family}/{target}"
    if not force:
        payload = _successful_payload(
            path,
            search_signature=_WORKER_SEARCH_SIGNATURE,
            source_signature=source_signature,
        )
        if payload is not None:
            return {
                "status": "skipped",
                "target": target_key,
                "candidate_seed": candidate_seed,
            }
    if _WORKER_PROJECTS is None or _WORKER_OVERSAMPLER is None:
        raise RuntimeError("Loss-seed-search worker was not initialized")

    print(
        f"LOSS_SEED_RUN target={target_key} candidate_seed={candidate_seed}",
        flush=True,
    )
    started_at = _utc_now()
    started = time.perf_counter()
    tail: deque[Dict[str, Any]] = deque(maxlen=_WORKER_TAIL_BATCH_COUNT)

    def collect(row: Mapping[str, Any]) -> None:
        tail.append({field: row[field] for field in LOSS_RECORD_FIELDS})

    try:
        seed_everything(candidate_seed)
        sources = tuple(
            sources_for_target(_WORKER_PROJECTS, family, target)
        )
        metadata = _WORKER_OVERSAMPLER.fit_training_only(
            sources,
            candidate_seed,
            loss_callback=collect,
        )
        if metadata.get("status") != "success":
            raise RuntimeError(
                f"IPGDO training did not produce losses: {metadata.get('status')}"
            )
        if len(tail) != _WORKER_TAIL_BATCH_COUNT:
            raise RuntimeError(
                f"Expected {_WORKER_TAIL_BATCH_COUNT} final batches, got {len(tail)}"
            )
        final_batches = list(tail)
        means = {
            field: _mean([float(row[field]) for row in final_batches])
            for field in LOSS_FIELDS
        }
        score_values = [
            float(row[_WORKER_SCORE_FIELD]) for row in final_batches
        ]
        diffusion = _mapping(metadata.get("diffusion"), "diffusion metadata")
        payload = {
            "schema_version": 1,
            "status": "success",
            "selection_protocol": "source-training-final-batch-loss",
            "target_labels_used": False,
            "target": target_key,
            "family": family,
            "project": target,
            "candidate_seed": int(candidate_seed),
            "score_field": _WORKER_SCORE_FIELD,
            "score": means[_WORKER_SCORE_FIELD],
            "score_std": _std(score_values),
            "final_batch_count": len(final_batches),
            "final_global_batch_start": int(final_batches[0]["global_batch"]),
            "final_global_batch_end": int(final_batches[-1]["global_batch"]),
            "mean_losses": means,
            "final_batches": final_batches,
            "source_projects": [source.key for source in sources],
            "source_data_signature": source_signature,
            "search_signature": _WORKER_SEARCH_SIGNATURE,
            "device": diffusion.get("device"),
            "amp_enabled": diffusion.get("amp_enabled"),
            "diffusion_configuration": diffusion.get("configuration"),
            "started_at": started_at,
            "completed_at": _utc_now(),
            "duration_seconds": time.perf_counter() - started,
        }
        _atomic_json(path, payload)
        print(
            f"LOSS_SEED_OK target={target_key} candidate_seed={candidate_seed} "
            f"tail_{_WORKER_SCORE_FIELD}_mean={payload['score']:.9f}",
            flush=True,
        )
        return {
            "status": "success",
            "target": target_key,
            "candidate_seed": candidate_seed,
            "score": payload["score"],
        }
    except Exception as exc:
        _atomic_json(
            path,
            {
                "schema_version": 1,
                "status": "failure",
                "selection_protocol": "source-training-final-batch-loss",
                "target_labels_used": False,
                "target": target_key,
                "candidate_seed": int(candidate_seed),
                "source_data_signature": source_signature,
                "search_signature": _WORKER_SEARCH_SIGNATURE,
                "started_at": started_at,
                "completed_at": _utc_now(),
                "duration_seconds": time.perf_counter() - started,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        print(
            f"LOSS_SEED_FAILURE target={target_key} candidate_seed={candidate_seed} "
            f"error={type(exc).__name__}: {exc}",
            flush=True,
        )
        return {
            "status": "failure",
            "target": target_key,
            "candidate_seed": candidate_seed,
            "error": str(exc),
        }


def _ranking_rows(
    *,
    output_dir: Path,
    family: str,
    target: str,
    candidate_seeds: Sequence[int],
    search_signature: str,
    source_signature: str,
    score_field: str,
) -> List[Dict[str, Any]]:
    target_key = f"{family}/{target}"
    rows: List[Dict[str, Any]] = []
    missing = []
    for seed in candidate_seeds:
        payload = _successful_payload(
            _run_path(output_dir, family, target, seed),
            search_signature=search_signature,
            source_signature=source_signature,
        )
        if payload is None:
            missing.append(seed)
            continue
        means = _mapping(payload.get("mean_losses"), "mean_losses")
        rows.append(
            {
                "target": target_key,
                "candidate_seed": int(seed),
                f"tail_{score_field}_mean": float(payload["score"]),
                f"tail_{score_field}_std": float(payload["score_std"]),
                **{
                    f"tail_{field}_mean": float(means[field])
                    for field in LOSS_FIELDS
                },
                "final_batch_count": int(payload["final_batch_count"]),
                "final_global_batch_start": int(
                    payload["final_global_batch_start"]
                ),
                "final_global_batch_end": int(payload["final_global_batch_end"]),
                "duration_seconds": float(payload["duration_seconds"]),
                "device": payload.get("device"),
                "amp_enabled": payload.get("amp_enabled"),
            }
        )
    if missing:
        preview = ", ".join(str(seed) for seed in missing[:10])
        raise RuntimeError(
            f"{target_key} has {len(missing)} incomplete candidate seeds: {preview}"
        )
    score_column = f"tail_{score_field}_mean"
    rows.sort(
        key=lambda row: (
            row[score_column],
            row["tail_noise_loss_mean"],
            row["tail_physics_loss_mean"],
            row["candidate_seed"],
        )
    )
    for position, row in enumerate(rows, start=1):
        row["selection_position"] = position
    return rows


def _write_rankings(
    output_dir: Path,
    target_key: str,
    rows: Sequence[Mapping[str, Any]],
) -> Path:
    path = _ranking_path(output_dir, target_key)
    fieldnames = ("selection_position",) + tuple(
        key for key in rows[0] if key != "selection_position"
    )
    _atomic_csv(path, fieldnames, rows)
    return path


class IPGDOLossSeedSearch:
    def __init__(self, config: LossSeedSearchConfig) -> None:
        self.config = config
        self.experiment = load_config(config.experiment_config)
        method_params = self.experiment.method_params(
            config.profile, config.method
        )
        oversampling = method_params.get("oversampling")
        if not isinstance(oversampling, Mapping):
            raise ValueError(
                f"{config.profile}/{config.method} has no oversampling configuration"
            )
        self.oversampling_config = dict(oversampling)
        self.search_signature = _search_signature(
            config, self.oversampling_config
        )
        self.projects: Optional[Mapping[str, Any]] = None

    def _source_context(
        self, family: str, target: str
    ) -> Tuple[Tuple[Any, ...], str]:
        if self.projects is None:
            raise RuntimeError("Projects have not been loaded")
        sources = tuple(sources_for_target(self.projects, family, target))
        return sources, source_data_signature(sources)

    def _build_winner_cache(
        self,
        *,
        output_dir: Path,
        family: str,
        target: str,
        best_seed: int,
        sources: Sequence[Any],
    ) -> Path:
        target_key = f"{family}/{target}"
        cache_path = (
            output_dir
            / "cache"
            / _safe_component(target_key)
            / f"seed_{best_seed}.npz"
        )
        try:
            load_augmentation_cache(
                cache_path,
                target_key=target_key,
                oversampling_seed=best_seed,
                oversampling_config=self.oversampling_config,
                sources=sources,
            )
            print(
                f"LOSS_SEED_CACHE_REUSE target={target_key} seed={best_seed}",
                flush=True,
            )
            return cache_path
        except (FileNotFoundError, KeyError, ValueError):
            pass
        print(
            f"LOSS_SEED_CACHE_BUILD target={target_key} seed={best_seed}",
            flush=True,
        )
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
        return cache_path

    def run(
        self,
        *,
        targets: Optional[Sequence[str]] = None,
        seed_count: Optional[int] = None,
        workers: Optional[int] = None,
        output_dir: Optional[Path] = None,
        force: bool = False,
        dry_run: bool = False,
        build_cache: Optional[bool] = None,
    ) -> Path:
        selected_targets = _selected_targets(self.experiment, targets or ())
        candidate_seeds = self.config.candidate_seeds
        if seed_count is not None:
            if seed_count < 1 or seed_count > len(candidate_seeds):
                raise ValueError(
                    f"seed_count must be between 1 and {len(candidate_seeds)}"
                )
            candidate_seeds = candidate_seeds[:seed_count]
        effective_workers = self.config.gpu_workers if workers is None else workers
        if effective_workers < 1:
            raise ValueError("workers must be positive")
        effective_output = (
            self.config.output_dir
            if output_dir is None
            else output_dir.resolve()
        )
        effective_build_cache = (
            self.config.build_winner_cache
            if build_cache is None
            else build_cache
        )
        effective_output.mkdir(parents=True, exist_ok=True)
        plan = {
            "schema_version": 1,
            "selection_protocol": "source-training-final-batch-loss",
            "target_labels_used": False,
            "targets": [
                f"{family}/{target}" for family, target in selected_targets
            ],
            "target_count": len(selected_targets),
            "candidate_seeds": list(candidate_seeds),
            "candidate_seed_count": len(candidate_seeds),
            "training_tasks": len(selected_targets) * len(candidate_seeds),
            "final_batch_count": self.config.final_batch_count,
            "score_field": self.config.score_field,
            "gpu_workers": effective_workers,
            "build_winner_cache": effective_build_cache,
            "experiment_config": str(self.config.experiment_config),
            "search_signature": self.search_signature,
        }
        _atomic_yaml(effective_output / "search_plan.yaml", plan)
        print(f"LOSS_SEED_SEARCH_PLAN {plan}", flush=True)
        if dry_run:
            return effective_output / "best_loss_seeds.yaml"

        self.projects = load_all_projects(self.experiment.data)
        contexts: Dict[str, Tuple[Tuple[Any, ...], str]] = {}
        for family, target in selected_targets:
            contexts[f"{family}/{target}"] = self._source_context(family, target)

        winners: Dict[str, Dict[str, Any]] = {}
        total_success = 0
        total_skipped = 0
        total_failures = 0
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=effective_workers,
            mp_context=context,
            initializer=_worker_init,
            initargs=(
                str(self.config.experiment_config),
                self.config.profile,
                self.config.method,
                self.config.final_batch_count,
                self.config.score_field,
                self.search_signature,
            ),
        ) as executor:
            for target_index, (family, target) in enumerate(
                selected_targets, start=1
            ):
                target_key = f"{family}/{target}"
                sources, source_signature = contexts[target_key]
                tasks = [
                    (
                        family,
                        target,
                        seed,
                        str(effective_output),
                        source_signature,
                        force,
                    )
                    for seed in candidate_seeds
                ]
                pending = []
                for task in tasks:
                    if force or _successful_payload(
                        _run_path(effective_output, family, target, task[2]),
                        search_signature=self.search_signature,
                        source_signature=source_signature,
                    ) is None:
                        pending.append(task)
                    else:
                        total_skipped += 1
                print(
                    f"LOSS_SEED_TARGET {target_index}/{len(selected_targets)} "
                    f"target={target_key} pending={len(pending)} "
                    f"skipped={len(tasks) - len(pending)}",
                    flush=True,
                )
                futures = [executor.submit(_run_one, task) for task in pending]
                target_failures = 0
                for future in as_completed(futures):
                    result = future.result()
                    if result["status"] == "success":
                        total_success += 1
                    elif result["status"] == "skipped":
                        total_skipped += 1
                    else:
                        total_failures += 1
                        target_failures += 1
                if target_failures:
                    raise RuntimeError(
                        f"{target_key} has {target_failures} failed candidates; "
                        "rerun the command to resume"
                    )
                ranking_rows = _ranking_rows(
                    output_dir=effective_output,
                    family=family,
                    target=target,
                    candidate_seeds=candidate_seeds,
                    search_signature=self.search_signature,
                    source_signature=source_signature,
                    score_field=self.config.score_field,
                )
                ranking_path = _write_rankings(
                    effective_output, target_key, ranking_rows
                )
                winner = dict(ranking_rows[0])
                winners[target_key] = {
                    "best_seed": int(winner["candidate_seed"]),
                    "ranking_file": os.path.relpath(
                        ranking_path, effective_output
                    ),
                    "source_projects": [source.key for source in sources],
                    "source_data_signature": source_signature,
                    "search_signature": self.search_signature,
                    "selection": winner,
                }
                _atomic_yaml(
                    effective_output / "best_loss_seeds.yaml",
                    {
                        **plan,
                        "updated_at": _utc_now(),
                        "completed_target_count": len(winners),
                        "targets": winners,
                    },
                )
                print(
                    f"LOSS_SEED_WINNER target={target_key} "
                    f"seed={winner['candidate_seed']} "
                    f"tail_{self.config.score_field}_mean="
                    f"{winner[f'tail_{self.config.score_field}_mean']:.9f}",
                    flush=True,
                )

        compatible_entries: Dict[str, Dict[str, Any]] = {}
        if effective_build_cache:
            for target_index, (family, target) in enumerate(
                selected_targets, start=1
            ):
                target_key = f"{family}/{target}"
                sources, _ = contexts[target_key]
                winner = winners[target_key]
                best_seed = int(winner["best_seed"])
                print(
                    f"LOSS_SEED_CACHE_TARGET {target_index}/{len(selected_targets)} "
                    f"target={target_key}",
                    flush=True,
                )
                cache_path = self._build_winner_cache(
                    output_dir=effective_output,
                    family=family,
                    target=target,
                    best_seed=best_seed,
                    sources=sources,
                )
                compatible_entries[target_key] = {
                    **winner,
                    "cache_file": os.path.relpath(cache_path, effective_output),
                }
                _atomic_yaml(
                    effective_output / "best_oversampling_seeds.yaml",
                    {
                        "schema_version": 1,
                        "selection_protocol": (
                            "source-training-final-batch-loss"
                        ),
                        "target_labels_used": False,
                        "method": self.config.method,
                        "profile": self.config.profile,
                        "candidate_seeds": list(candidate_seeds),
                        "final_batch_count": self.config.final_batch_count,
                        "score_field": self.config.score_field,
                        "experiment_config": str(
                            self.config.experiment_config
                        ),
                        "updated_at": _utc_now(),
                        "targets": compatible_entries,
                    },
                )

        best_rows = []
        for target_key, entry in winners.items():
            selection = entry["selection"]
            best_rows.append(
                {
                    "target": target_key,
                    "best_seed": entry["best_seed"],
                    f"tail_{self.config.score_field}_mean": selection[
                        f"tail_{self.config.score_field}_mean"
                    ],
                    "tail_noise_loss_mean": selection[
                        "tail_noise_loss_mean"
                    ],
                    "tail_physics_loss_mean": selection[
                        "tail_physics_loss_mean"
                    ],
                    "ranking_file": entry["ranking_file"],
                    "cache_file": (
                        compatible_entries.get(target_key, {}).get("cache_file")
                    ),
                }
            )
        if best_rows:
            _atomic_csv(
                effective_output / "summary" / "best_seeds.csv",
                tuple(best_rows[0]),
                best_rows,
            )
        manifest = {
            "status": "success",
            "completed_at": _utc_now(),
            **plan,
            "successful_tasks_in_this_run": total_success,
            "skipped_tasks": total_skipped,
            "failed_tasks_in_this_run": total_failures,
            "best_loss_seed_map": str(
                effective_output / "best_loss_seeds.yaml"
            ),
            "compatible_cache_seed_map": (
                str(effective_output / "best_oversampling_seeds.yaml")
                if effective_build_cache
                else None
            ),
        }
        _atomic_json(effective_output / "run_manifest.json", manifest)
        print(
            f"LOSS_SEED_SEARCH_COMPLETE targets={len(winners)} "
            f"seed_map={manifest['best_loss_seed_map']}",
            flush=True,
        )
        return effective_output / "best_loss_seeds.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select the lowest-tail-loss IPGDO seed for every CPDP target"
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--target", action="append")
    parser.add_argument(
        "--seed-count",
        type=int,
        help="Use only the first N configured candidate seeds",
    )
    parser.add_argument("--workers", type=int)
    parser.add_argument("--output")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    search = IPGDOLossSeedSearch(
        load_loss_seed_search_config(Path(args.config))
    )
    search.run(
        targets=args.target,
        seed_count=args.seed_count,
        workers=args.workers,
        output_dir=Path(args.output) if args.output else None,
        force=args.force,
        dry_run=args.dry_run,
        build_cache=False if args.no_cache else None,
    )
    return 0


__all__ = [
    "IPGDOLossSeedSearch",
    "LossSeedSearchConfig",
    "load_loss_seed_search_config",
    "main",
]
