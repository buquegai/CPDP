from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import time
import traceback
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml

from cpdp.config import ExperimentConfig, load_config
from cpdp.data import load_all_projects, sources_for_target
from cpdp.metrics import METRIC_NAMES, evaluate_predictions
from cpdp.oversampling.config import DiffusionConfig, KNNFilterConfig
from cpdp.oversampling.diffusion import ProjectConditionedTabularDDPM
from cpdp.oversampling.manifold_filter import DefectManifoldFilter
from cpdp.oversampling.preprocessing import RobustAffineScaler
from cpdp.oversampling.types import AugmentationResult, freeze_array
from cpdp.oversampling_validation.classifier import predict_validation_target
from cpdp.oversampling_validation.config import ClassifierConfig
from cpdp.randomness import seed_everything
from cpdp.types import LabeledProject, ProjectFeatures


EXPERIMENT_VERSION = "knn-selection-ablation-v1"
SELECTION_STRATEGIES = (
    "shared_pool_raw_random",
    "knn_interval_random",
    "knn_interval_median",
)


@dataclass(frozen=True)
class KNNSelectionConfig:
    """Configuration isolated from the current IPGDO production pipeline."""

    device: str
    target_minority_ratio: float
    candidate_batch_multiplier: float
    max_generation_rounds: int
    diffusion: DiffusionConfig
    knn: KNNFilterConfig

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "KNNSelectionConfig":
        diffusion_values = values.get("diffusion", {})
        knn_values = values.get("knn", {})
        if not isinstance(diffusion_values, Mapping):
            raise TypeError("diffusion must be a mapping")
        if not isinstance(knn_values, Mapping):
            raise TypeError("knn must be a mapping")
        diffusion = DiffusionConfig(
            **{
                key: value
                for key, value in diffusion_values.items()
                if key in DiffusionConfig.__dataclass_fields__
            }
        )
        knn = KNNFilterConfig(**knn_values)
        config = cls(
            device=str(values.get("device", "auto")),
            target_minority_ratio=float(values.get("target_minority_ratio", 1.0)),
            candidate_batch_multiplier=float(
                values.get("candidate_batch_multiplier", 2.0)
            ),
            max_generation_rounds=int(values.get("max_generation_rounds", 100)),
            diffusion=diffusion,
            knn=knn,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
        if self.target_minority_ratio <= 0:
            raise ValueError("target_minority_ratio must be positive")
        if self.candidate_batch_multiplier < 1.0:
            raise ValueError("candidate_batch_multiplier must be at least 1")
        if self.max_generation_rounds < 1:
            raise ValueError("max_generation_rounds must be at least 1")
        self.diffusion.validate()
        self.knn.validate()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _expand_path(value: str, base: Path) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(value))
    path = Path(expanded)
    return path if path.is_absolute() else (base / path).resolve()


def _deep_merge(
    base: Mapping[str, Any], overrides: Mapping[str, Any]
) -> Dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _derived_seed(seed: int, *parts: object) -> int:
    payload = ":".join((str(seed), *(str(part) for part in parts))).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _named_hash(items: Iterable[Tuple[str, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for name, values in items:
        digest.update(name.encode("utf-8"))
        digest.update(array_sha256(values).encode("ascii"))
    return digest.hexdigest()


@dataclass(frozen=True)
class AblationConfig:
    source_path: Path
    experiment: ExperimentConfig
    output_dir: Path
    profile: str
    method: str
    seeds: Tuple[int, ...]
    targets: Tuple[str, ...]
    oversampling: KNNSelectionConfig
    predictor: Mapping[str, Any]

    def effective_mapping(self) -> Dict[str, Any]:
        return {
            "experiment_version": EXPERIMENT_VERSION,
            "experiment_config": str(self.experiment.source_path),
            "profile": self.profile,
            "method": self.method,
            "seeds": list(self.seeds),
            "targets": list(self.targets),
            "selection_strategies": list(SELECTION_STRATEGIES),
            "oversampling": asdict(self.oversampling),
            "predictor": deepcopy(dict(self.predictor)),
            "metrics": list(METRIC_NAMES),
        }

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.effective_mapping(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]


def load_ablation_config(path: Path) -> AblationConfig:
    source_path = path.resolve()
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping) or raw.get("schema_version") != 1:
        raise ValueError("Unsupported KNN selection ablation schema_version")

    experiment_path = _expand_path(
        str(raw.get("experiment_config", "experiment.yaml")), source_path.parent
    )
    experiment = load_config(experiment_path)
    profile = str(raw.get("profile", "full"))
    method = str(raw.get("method", "ipgdo_kan_jepa"))
    method_params = experiment.method_params(profile, method)
    base_oversampling = method_params.get("oversampling")
    predictor = method_params.get("predictor")
    if not isinstance(base_oversampling, Mapping):
        raise ValueError(f"{profile}/{method} must define oversampling")
    if not isinstance(predictor, Mapping):
        raise ValueError(f"{profile}/{method} must define predictor")

    configured_oversampling = raw.get("oversampling", {})
    if not isinstance(configured_oversampling, Mapping):
        raise TypeError("oversampling must be a mapping")
    oversampling = KNNSelectionConfig.from_mapping(
        _deep_merge(base_oversampling, configured_oversampling)
    )

    raw_seeds = raw.get("seeds", "experiment")
    seeds = (
        experiment.seeds
        if isinstance(raw_seeds, str) and raw_seeds.lower() == "experiment"
        else tuple(int(seed) for seed in raw_seeds)
    )
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a non-empty unique sequence")

    available_targets = tuple(
        f"{family}/{target}" for family, target in experiment.targets()
    )
    raw_targets = raw.get("targets", "all")
    targets = (
        available_targets
        if isinstance(raw_targets, str) and raw_targets.lower() == "all"
        else tuple(str(target).replace("\\", "/") for target in raw_targets)
    )
    unknown_targets = sorted(set(targets) - set(available_targets))
    if unknown_targets:
        raise KeyError("Unknown targets: " + ", ".join(unknown_targets))
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("targets must be a non-empty unique sequence")

    project_root = source_path.parent.parent
    return AblationConfig(
        source_path=source_path,
        experiment=experiment,
        output_dir=_expand_path(
            str(raw.get("output_dir", "outputs/knn_selection_ablation")),
            project_root,
        ),
        profile=profile,
        method=method,
        seeds=tuple(int(seed) for seed in seeds),
        targets=targets,
        oversampling=oversampling,
        predictor=deepcopy(dict(predictor)),
    )


@dataclass(frozen=True)
class ProjectCandidatePool:
    project_key: str
    requested: int
    candidates: np.ndarray
    accepted: np.ndarray
    quality_scores: np.ndarray
    mean_distances: np.ndarray
    metadata: Mapping[str, Any]

    @property
    def candidate_sha256(self) -> str:
        return array_sha256(self.candidates)

    @property
    def accepted_sha256(self) -> str:
        return _named_hash(
            (
                ("candidates", self.candidates[self.accepted]),
                ("quality_scores", self.quality_scores[self.accepted]),
            )
        )


@dataclass(frozen=True)
class SharedCandidatePools:
    sources: Tuple[LabeledProject, ...]
    project_pools: Tuple[ProjectCandidatePool, ...]
    diffusion_metadata: Mapping[str, Any]
    generation_seconds: float

    @property
    def candidate_pool_sha256(self) -> str:
        return _named_hash(
            (pool.project_key, pool.candidates) for pool in self.project_pools
        )


class SharedCandidatePoolBuilder:
    """Train one DDPM and retain one immutable candidate pool per source project."""

    def __init__(self, config: KNNSelectionConfig) -> None:
        self.config = config

    def build(
        self, sources: Sequence[LabeledProject], seed: int
    ) -> SharedCandidatePools:
        started = time.perf_counter()
        projects = tuple(sources)
        self._validate_sources(projects)
        requested = tuple(self._requested_count(project) for project in projects)
        if sum(requested) == 0:
            empty_pools = tuple(
                ProjectCandidatePool(
                    project_key=project.key,
                    requested=0,
                    candidates=freeze_array(
                        np.empty((0, project.features.n_features)), np.float64
                    ),
                    accepted=freeze_array(np.empty(0), bool),
                    quality_scores=freeze_array(np.empty(0), np.float64),
                    mean_distances=freeze_array(np.empty(0), np.float64),
                    metadata={"reason": "not_requested", "rounds": []},
                )
                for project in projects
            )
            return SharedCandidatePools(
                sources=projects,
                project_pools=empty_pools,
                diffusion_metadata={"enabled": False, "reason": "not_requested"},
                generation_seconds=time.perf_counter() - started,
            )

        defect_batches = tuple(
            project.features.values[project.labels == 1] for project in projects
        )
        project_scalers = tuple(
            RobustAffineScaler.fit(defects) for defects in defect_batches
        )
        normalized_defects = np.vstack(
            [
                np.clip(
                    scaler.transform(defects),
                    -self.config.diffusion.normalized_clip,
                    self.config.diffusion.normalized_clip,
                )
                for defects, scaler in zip(defect_batches, project_scalers)
            ]
        ).astype(np.float32)
        project_ids = np.concatenate(
            [
                np.full(len(defects), index, dtype=np.int64)
                for index, defects in enumerate(defect_batches)
            ]
        )
        diffusion = ProjectConditionedTabularDDPM(
            input_dim=normalized_defects.shape[1],
            project_count=len(projects),
            config=self.config.diffusion,
            device=self.config.device,
        )
        diffusion_metadata = diffusion.fit(normalized_defects, project_ids, seed)

        project_pools = []
        for project_id, (project, needed, scaler) in enumerate(
            zip(projects, requested, project_scalers)
        ):
            if needed == 0:
                project_pools.append(
                    ProjectCandidatePool(
                        project_key=project.key,
                        requested=0,
                        candidates=freeze_array(
                            np.empty((0, project.features.n_features)), np.float64
                        ),
                        accepted=freeze_array(np.empty(0), bool),
                        quality_scores=freeze_array(np.empty(0), np.float64),
                        mean_distances=freeze_array(np.empty(0), np.float64),
                        metadata={"reason": "not_requested", "rounds": []},
                    )
                )
                continue
            project_pools.append(
                self._generate_project_pool(
                    diffusion, scaler, project, project_id, needed
                )
            )
        del diffusion
        _release_cuda_cache()
        return SharedCandidatePools(
            sources=projects,
            project_pools=tuple(project_pools),
            diffusion_metadata=diffusion_metadata,
            generation_seconds=time.perf_counter() - started,
        )

    def _generate_project_pool(
        self,
        diffusion: ProjectConditionedTabularDDPM,
        scaler: RobustAffineScaler,
        project: LabeledProject,
        project_id: int,
        needed: int,
    ) -> ProjectCandidatePool:
        defects = project.features.values[project.labels == 1]
        knn_filter = DefectManifoldFilter(defects, self.config.knn)
        candidate_batches = []
        accepted_batches = []
        quality_batches = []
        distance_batches = []
        rounds = []
        accepted_count = 0

        for round_number in range(1, self.config.max_generation_rounds + 1):
            remaining = needed - accepted_count
            if remaining <= 0:
                break
            candidate_count = max(
                1,
                int(math.ceil(remaining * self.config.candidate_batch_multiplier)),
            )
            normalized = diffusion.sample_defects(candidate_count, project_id)
            candidates = np.asarray(
                scaler.inverse_transform(normalized), dtype=np.float64
            )
            evaluation = knn_filter.evaluate(candidates)
            candidate_batches.append(candidates)
            accepted_batches.append(np.asarray(evaluation.accepted, dtype=bool))
            quality_batches.append(
                np.asarray(evaluation.quality_scores, dtype=np.float64)
            )
            distance_batches.append(
                np.asarray(evaluation.mean_distances, dtype=np.float64)
            )
            accepted_count += int(np.sum(evaluation.accepted))
            rounds.append({"round": round_number, **evaluation.metadata})
            print(
                "KNN_ABLATION_POOL "
                f"project={project.key} round={round_number}/"
                f"{self.config.max_generation_rounds} "
                f"generated={candidate_count} accepted_batch="
                f"{int(np.sum(evaluation.accepted))} "
                f"accepted_total={accepted_count}/{needed}",
                flush=True,
            )

        candidates = np.vstack(candidate_batches)
        accepted = np.concatenate(accepted_batches)
        quality_scores = np.concatenate(quality_batches)
        mean_distances = np.concatenate(distance_batches)
        if accepted_count < needed:
            raise RuntimeError(
                f"Could not obtain {needed} KNN-qualified samples for {project.key}; "
                f"accepted {accepted_count} from {len(candidates)} candidates"
            )
        return ProjectCandidatePool(
            project_key=project.key,
            requested=needed,
            candidates=freeze_array(candidates, np.float64),
            accepted=freeze_array(accepted, bool),
            quality_scores=freeze_array(quality_scores, np.float64),
            mean_distances=freeze_array(mean_distances, np.float64),
            metadata={
                "generated_candidates": int(len(candidates)),
                "knn_accepted_candidates": int(accepted_count),
                "generation_rounds": int(len(rounds)),
                "knn": {**knn_filter.metadata(), "rounds": rounds},
            },
        )

    def _requested_count(self, project: LabeledProject) -> int:
        clean = int(np.sum(project.labels == 0))
        defects = int(np.sum(project.labels == 1))
        return max(
            0,
            int(math.ceil(clean * self.config.target_minority_ratio)) - defects,
        )

    @staticmethod
    def _validate_sources(projects: Tuple[LabeledProject, ...]) -> None:
        if not projects:
            raise ValueError("At least one source project is required")
        family = projects[0].features.family
        schema = projects[0].features.feature_names
        keys = set()
        for project in projects:
            if project.key in keys:
                raise ValueError(f"Duplicate source project: {project.key}")
            keys.add(project.key)
            if project.features.family != family:
                raise ValueError("All source projects must belong to one family")
            if project.features.feature_names != schema:
                raise ValueError("All source projects must share one feature schema")
            if set(np.unique(project.labels).tolist()) != {0, 1}:
                raise ValueError(f"Source project lacks both classes: {project.key}")
            if int(np.sum(project.labels == 1)) < 2:
                raise ValueError(
                    f"Source project needs at least two defects: {project.key}"
                )


def select_project_candidates(
    pool: ProjectCandidatePool, strategy: str, seed: int
) -> Tuple[np.ndarray, Mapping[str, Any]]:
    if strategy not in SELECTION_STRATEGIES:
        raise KeyError(f"Unknown selection strategy: {strategy}")
    if pool.requested == 0:
        empty = np.empty((0, pool.candidates.shape[1]), dtype=np.float64)
        return empty, {
            "strategy": strategy,
            "selected": 0,
            "selected_sha256": array_sha256(empty),
        }

    if strategy == "shared_pool_raw_random":
        eligible_indices = np.arange(len(pool.candidates), dtype=np.int64)
        ranking_rule = "uniform random without replacement from the shared raw pool"
    else:
        eligible_indices = np.flatnonzero(pool.accepted)
        ranking_rule = (
            "uniform random without replacement inside the KNN interval"
            if strategy == "knn_interval_random"
            else "smallest normalized distance from the real-defect KNN median"
        )
    if len(eligible_indices) < pool.requested:
        raise RuntimeError(
            f"{strategy} has only {len(eligible_indices)} eligible candidates for "
            f"{pool.project_key}, but {pool.requested} are required"
        )

    if strategy == "knn_interval_median":
        eligible_scores = pool.quality_scores[eligible_indices]
        order = np.argsort(eligible_scores, kind="stable")[: pool.requested]
        selected_indices = eligible_indices[order]
    else:
        random = np.random.default_rng(
            _derived_seed(seed, pool.project_key, strategy, "selection")
        )
        selected_indices = np.sort(
            random.choice(
                eligible_indices, size=pool.requested, replace=False
            ).astype(np.int64)
        )

    selected = np.asarray(pool.candidates[selected_indices], dtype=np.float64)
    selected_scores = pool.quality_scores[selected_indices]
    finite_scores = selected_scores[np.isfinite(selected_scores)]
    metadata: Dict[str, Any] = {
        "strategy": strategy,
        "ranking_rule": ranking_rule,
        "eligible": int(len(eligible_indices)),
        "selected": int(len(selected)),
        "selected_indices_sha256": array_sha256(selected_indices),
        "selected_sha256": array_sha256(selected),
    }
    if len(finite_scores):
        metadata.update(
            {
                "selected_quality_min": float(np.min(finite_scores)),
                "selected_quality_median": float(np.median(finite_scores)),
                "selected_quality_max": float(np.max(finite_scores)),
            }
        )
    return selected, metadata


def build_strategy_augmentation(
    shared: SharedCandidatePools, strategy: str, seed: int
) -> AugmentationResult:
    augmented = []
    diagnostics = []
    selected_arrays = []
    for project, pool in zip(shared.sources, shared.project_pools):
        selected, selection_metadata = select_project_candidates(pool, strategy, seed)
        augmented.append(_append_synthetic(project, selected))
        selected_arrays.append((project.key, selected))
        diagnostics.append(
            {
                "project": project.key,
                "requested_synthetic": int(pool.requested),
                "generated_candidates": int(len(pool.candidates)),
                "knn_accepted_candidates": int(np.sum(pool.accepted)),
                "candidate_pool_sha256": pool.candidate_sha256,
                "knn_accepted_pool_sha256": pool.accepted_sha256,
                **selection_metadata,
            }
        )
    return AugmentationResult(
        projects=tuple(augmented),
        metadata={
            "name": strategy,
            "strictly_paired": True,
            "target_data_used": False,
            "shared_candidate_pool_sha256": shared.candidate_pool_sha256,
            "selected_pool_sha256": _named_hash(selected_arrays),
            "selection": diagnostics,
            "total_synthetic_selected": int(
                sum(item["selected"] for item in diagnostics)
            ),
        },
    )


def _append_synthetic(
    project: LabeledProject, synthetic_values: np.ndarray
) -> LabeledProject:
    values = np.asarray(synthetic_values, dtype=np.float64)
    if len(values) == 0:
        return project
    combined = np.vstack((project.features.values, values))
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
        values=freeze_array(combined, np.float64),
        feature_names=project.features.feature_names,
        effort=freeze_array(effort, np.float64),
        effort_name=project.features.effort_name,
        row_ids=freeze_array(row_ids, np.int64),
        cleaning=cleaning,
    )
    return LabeledProject(
        features=features,
        labels=freeze_array(labels, np.int64),
    )


class AblationStore:
    def __init__(self, output_dir: Path, fingerprint: str) -> None:
        self.root = output_dir / fingerprint
        self.runs_dir = self.root / "runs"
        self.summary_dir = self.root / "summary"

    def task_path(self, target_key: str, seed: int) -> Path:
        family, target = target_key.split("/", 1)
        return self.runs_dir / family / target / f"seed_{seed}.json"

    @staticmethod
    def load(path: Path) -> Dict[str, Any]:
        if not path.is_file():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        return loaded if isinstance(loaded, dict) else {}

    @staticmethod
    def save(path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(path)

    @staticmethod
    def is_complete(path: Path) -> bool:
        record = AblationStore.load(path)
        strategies = record.get("strategies", {})
        return record.get("status") == "success" and all(
            isinstance(strategies.get(name), Mapping)
            and strategies[name].get("status") == "success"
            for name in SELECTION_STRATEGIES
        )

    def write_manifest(self, config: AblationConfig) -> Path:
        path = self.root / "manifest.json"
        self.save(
            path,
            {
                "fingerprint": config.fingerprint,
                "created_at": _utc_now(),
                "configuration": config.effective_mapping(),
            },
        )
        return path

    def summarize(self) -> Mapping[str, Path]:
        records = [
            self.load(path) for path in sorted(self.runs_dir.glob("**/seed_*.json"))
        ]
        records = [record for record in records if record]
        if not records:
            raise RuntimeError("No KNN selection ablation results are available")
        self.summary_dir.mkdir(parents=True, exist_ok=True)

        metric_rows = []
        audit_rows = []
        for record in records:
            shared = record.get("shared_candidate_pool", {})
            projects = shared.get("projects", ())
            generated = sum(
                int(item.get("generated_candidates", 0)) for item in projects
            )
            accepted = sum(
                int(item.get("knn_accepted_candidates", 0)) for item in projects
            )
            for strategy, result in record.get("strategies", {}).items():
                metrics = result.get("metrics", {})
                row = {
                    "fingerprint": record.get("fingerprint"),
                    "family": record.get("family"),
                    "target": record.get("target"),
                    "target_key": record.get("target_key"),
                    "seed": record.get("seed"),
                    "strategy": strategy,
                    "status": result.get("status"),
                    "shared_candidate_pool_sha256": result.get(
                        "shared_candidate_pool_sha256"
                    ),
                    "selected_pool_sha256": result.get("selected_pool_sha256"),
                    "generated_candidates": generated,
                    "knn_accepted_candidates": accepted,
                    "selected_synthetic": result.get("selected_synthetic"),
                    "generation_seconds": shared.get("generation_seconds"),
                    "prediction_seconds": result.get("prediction_seconds"),
                    "threshold": result.get("threshold"),
                    **{metric: metrics.get(metric) for metric in METRIC_NAMES},
                }
                metric_rows.append(row)
                selection_by_project = {
                    item.get("project"): item
                    for item in result.get("selection", ())
                }
                for project in projects:
                    selection = selection_by_project.get(project.get("project"), {})
                    audit_rows.append(
                        {
                            "target_key": record.get("target_key"),
                            "seed": record.get("seed"),
                            "strategy": strategy,
                            "source_project": project.get("project"),
                            "shared_candidate_pool_sha256": result.get(
                                "shared_candidate_pool_sha256"
                            ),
                            "candidate_pool_sha256": project.get(
                                "candidate_pool_sha256"
                            ),
                            "knn_accepted_pool_sha256": project.get(
                                "knn_accepted_pool_sha256"
                            ),
                            "generated_candidates": project.get(
                                "generated_candidates"
                            ),
                            "knn_accepted_candidates": project.get(
                                "knn_accepted_candidates"
                            ),
                            "selected": selection.get("selected"),
                            "selected_sha256": selection.get("selected_sha256"),
                        }
                    )

        run_metrics_path = self.summary_dir / "target_seed_metrics.csv"
        _write_csv(run_metrics_path, metric_rows)
        audit_path = self.summary_dir / "candidate_pool_audit.csv"
        _write_csv(audit_path, audit_rows)

        successful = [row for row in metric_rows if row["status"] == "success"]
        target_groups: Dict[Tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
        for row in successful:
            target_groups[(str(row["target_key"]), str(row["strategy"]))].append(row)
        target_rows = []
        for (target_key, strategy), rows in sorted(target_groups.items()):
            family, target = target_key.split("/", 1)
            target_rows.append(
                {
                    "family": family,
                    "target": target,
                    "target_key": target_key,
                    "strategy": strategy,
                    "completed_seeds": len(rows),
                    **{
                        metric: _finite_mean(row.get(metric) for row in rows)
                        for metric in METRIC_NAMES
                    },
                }
            )
        target_metrics_path = self.summary_dir / "target_metrics_mean.csv"
        _write_csv(target_metrics_path, target_rows)

        macro_groups: Dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in target_rows:
            macro_groups[str(row["strategy"])].append(row)
        macro_rows = []
        for strategy, rows in sorted(macro_groups.items()):
            macro_rows.append(
                {
                    "strategy": strategy,
                    "completed_targets": len(rows),
                    "minimum_completed_seeds": min(
                        int(row["completed_seeds"]) for row in rows
                    ),
                    **{
                        metric: _finite_mean(row.get(metric) for row in rows)
                        for metric in METRIC_NAMES
                    },
                }
            )
        macro_path = self.summary_dir / "macro_metrics.csv"
        _write_csv(macro_path, macro_rows)
        return {
            "target_seed_metrics": run_metrics_path,
            "target_metrics_mean": target_metrics_path,
            "macro_metrics": macro_path,
            "candidate_pool_audit": audit_path,
        }


def _finite_mean(values: Iterable[Any]) -> float:
    numeric = np.asarray(
        [float(value) for value in values if value is not None], dtype=np.float64
    )
    numeric = numeric[np.isfinite(numeric)]
    return float(np.mean(numeric)) if len(numeric) else float("nan")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _pool_metadata(shared: SharedCandidatePools) -> Dict[str, Any]:
    return {
        "strictly_shared": True,
        "sha256": shared.candidate_pool_sha256,
        "generation_seconds": float(shared.generation_seconds),
        "diffusion_training": deepcopy(dict(shared.diffusion_metadata)),
        "projects": [
            {
                "project": pool.project_key,
                "requested_synthetic": int(pool.requested),
                "generated_candidates": int(len(pool.candidates)),
                "knn_accepted_candidates": int(np.sum(pool.accepted)),
                "candidate_pool_sha256": pool.candidate_sha256,
                "knn_accepted_pool_sha256": pool.accepted_sha256,
                **deepcopy(dict(pool.metadata)),
            }
            for pool in shared.project_pools
        ],
    }


def _release_cuda_cache() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@dataclass(frozen=True)
class RunReport:
    fingerprint: str
    selected: int
    attempted: int
    succeeded: int
    failed: int
    skipped: int
    output_root: Path


class KNNSelectionAblationRunner:
    def __init__(self, config: AblationConfig) -> None:
        self.config = config
        self.store = AblationStore(config.output_dir, config.fingerprint)

    def run(
        self,
        *,
        targets: Optional[Sequence[str]] = None,
        seeds: Optional[Sequence[int]] = None,
        max_tasks: Optional[int] = None,
        force: bool = False,
        fail_fast: bool = False,
    ) -> RunReport:
        selected_targets = self._select_targets(targets)
        selected_seeds = self._select_seeds(seeds)
        tasks = [
            (target_key, seed)
            for target_key in selected_targets
            for seed in selected_seeds
        ]
        pending = []
        skipped = 0
        for target_key, seed in tasks:
            path = self.store.task_path(target_key, seed)
            if not force and self.store.is_complete(path):
                skipped += 1
            else:
                pending.append((target_key, seed))
        if max_tasks is not None:
            if max_tasks < 1:
                raise ValueError("max_tasks must be positive")
            pending = pending[:max_tasks]

        self.store.write_manifest(self.config)
        if not pending:
            try:
                self.store.summarize()
            except RuntimeError:
                pass
            return RunReport(
                fingerprint=self.config.fingerprint,
                selected=len(tasks),
                attempted=0,
                succeeded=0,
                failed=0,
                skipped=skipped,
                output_root=self.store.root,
            )

        projects = load_all_projects(self.config.experiment.data)
        classifier = ClassifierConfig(
            kind="formal_kan_jepa_tabr",
            method=self.config.method,
            predictor=self.config.predictor,
        )
        classifier.validate()
        succeeded = 0
        failed = 0
        for index, (target_key, seed) in enumerate(pending, start=1):
            print(
                f"KNN_ABLATION task={index}/{len(pending)} "
                f"target={target_key} seed={seed}",
                flush=True,
            )
            try:
                target = projects[target_key]
                family, target_name = target_key.split("/", 1)
                source_projects = tuple(
                    sources_for_target(projects, family, target_name)
                )
                self._run_task(
                    source_projects,
                    target,
                    classifier,
                    int(seed),
                    force=force,
                )
                succeeded += 1
            except Exception as exc:
                failed += 1
                path = self.store.task_path(target_key, seed)
                record = self.store.load(path)
                record.update(
                    {
                        "fingerprint": self.config.fingerprint,
                        "target_key": target_key,
                        "seed": int(seed),
                        "status": "failure",
                        "failed_at": _utc_now(),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                self.store.save(path, record)
                print(
                    f"KNN_ABLATION_FAILURE target={target_key} seed={seed} "
                    f"error={type(exc).__name__}: {exc}",
                    flush=True,
                )
                if fail_fast:
                    raise
            try:
                self.store.summarize()
            except RuntimeError:
                pass
        return RunReport(
            fingerprint=self.config.fingerprint,
            selected=len(tasks),
            attempted=len(pending),
            succeeded=succeeded,
            failed=failed,
            skipped=skipped,
            output_root=self.store.root,
        )

    def _run_task(
        self,
        sources: Sequence[LabeledProject],
        target: LabeledProject,
        classifier: ClassifierConfig,
        seed: int,
        *,
        force: bool,
    ) -> None:
        task_started = time.perf_counter()
        path = self.store.task_path(target.key, seed)
        previous = {} if force else self.store.load(path)
        seed_everything(seed)
        shared = SharedCandidatePoolBuilder(self.config.oversampling).build(
            sources, seed
        )
        pool_metadata = _pool_metadata(shared)
        previous_hash = previous.get("shared_candidate_pool", {}).get("sha256")
        reset_reason = None
        if previous_hash and previous_hash != shared.candidate_pool_sha256:
            reset_reason = {
                "at": _utc_now(),
                "reason": "shared_candidate_pool_hash_changed_on_resume",
                "previous_sha256": previous_hash,
                "current_sha256": shared.candidate_pool_sha256,
            }
            previous = {}

        family, target_name = target.key.split("/", 1)
        record: Dict[str, Any] = {
            **previous,
            "fingerprint": self.config.fingerprint,
            "experiment_version": EXPERIMENT_VERSION,
            "family": family,
            "target": target_name,
            "target_key": target.key,
            "seed": int(seed),
            "status": "running",
            "started_at": previous.get("started_at", _utc_now()),
            "updated_at": _utc_now(),
            "strict_pairing": {
                "one_diffusion_fit_per_target_seed": True,
                "same_candidate_pool_for_all_strategies": True,
                "same_predictor_configuration": True,
                "same_predictor_seed": True,
            },
            "shared_candidate_pool": pool_metadata,
            "strategies": dict(previous.get("strategies", {})),
        }
        if reset_reason is not None:
            record["resume_reset"] = reset_reason
        self.store.save(path, record)

        for strategy in SELECTION_STRATEGIES:
            existing = record["strategies"].get(strategy, {})
            if (
                not force
                and existing.get("status") == "success"
                and existing.get("shared_candidate_pool_sha256")
                == shared.candidate_pool_sha256
            ):
                print(
                    f"KNN_ABLATION_SKIP target={target.key} seed={seed} "
                    f"strategy={strategy}",
                    flush=True,
                )
                continue
            augmentation = build_strategy_augmentation(shared, strategy, seed)
            prediction_started = time.perf_counter()
            seed_everything(seed)
            prediction = predict_validation_target(
                augmentation.projects,
                target,
                classifier,
                seed,
            )
            if prediction.evaluated_row_ids is not None and not np.array_equal(
                prediction.evaluated_row_ids, target.features.row_ids
            ):
                raise ValueError(
                    f"Predictor changed target row order for {target.key}"
                )
            metrics = evaluate_predictions(
                target.labels,
                prediction.scores,
                target.features.effort,
                prediction.threshold,
            )
            prediction_seconds = time.perf_counter() - prediction_started
            result = {
                "status": "success",
                "completed_at": _utc_now(),
                "shared_candidate_pool_sha256": shared.candidate_pool_sha256,
                "selected_pool_sha256": augmentation.metadata[
                    "selected_pool_sha256"
                ],
                "selected_synthetic": augmentation.metadata[
                    "total_synthetic_selected"
                ],
                "selection": augmentation.metadata["selection"],
                "threshold": float(prediction.threshold),
                "metrics": metrics,
                "prediction_seconds": float(prediction_seconds),
                "predictor": deepcopy(dict(prediction.metadata)),
            }
            record["strategies"][strategy] = result
            record["updated_at"] = _utc_now()
            self.store.save(path, record)
            print(
                f"KNN_ABLATION_RESULT target={target.key} seed={seed} "
                f"strategy={strategy} auc={metrics['auc']:.6f} "
                f"auprc={metrics['auprc']:.6f} mcc={metrics['mcc']:.6f} "
                f"f1={metrics['f1']:.6f} gmean={metrics['gmean']:.6f} "
                f"popt20={metrics['popt20']:.6f} "
                f"seconds={prediction_seconds:.1f}",
                flush=True,
            )
            del prediction, augmentation
            _release_cuda_cache()

        hashes = {
            result.get("shared_candidate_pool_sha256")
            for result in record["strategies"].values()
            if result.get("status") == "success"
        }
        if hashes != {shared.candidate_pool_sha256}:
            raise RuntimeError(
                f"Strict pairing audit failed for {target.key}, seed {seed}"
            )
        record.update(
            {
                "status": "success",
                "completed_at": _utc_now(),
                "updated_at": _utc_now(),
                "task_seconds": float(time.perf_counter() - task_started),
            }
        )
        self.store.save(path, record)
        del shared
        _release_cuda_cache()

    def _select_targets(
        self, requested: Optional[Sequence[str]]
    ) -> Tuple[str, ...]:
        if not requested:
            return self.config.targets
        normalized = tuple(str(value).replace("\\", "/") for value in requested)
        unknown = sorted(set(normalized) - set(self.config.targets))
        if unknown:
            raise KeyError("Targets are not enabled: " + ", ".join(unknown))
        return tuple(dict.fromkeys(normalized))

    def _select_seeds(
        self, requested: Optional[Sequence[int]]
    ) -> Tuple[int, ...]:
        if not requested:
            return self.config.seeds
        normalized = tuple(int(seed) for seed in requested)
        unknown = sorted(set(normalized) - set(self.config.seeds))
        if unknown:
            raise ValueError(f"Seeds are not enabled: {unknown}")
        return tuple(dict.fromkeys(normalized))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strictly paired KNN candidate-selection ablation"
    )
    parser.add_argument(
        "command", choices=("run", "summarize"), nargs="?", default="run"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/knn_selection_ablation.yaml"),
    )
    parser.add_argument("--targets", nargs="*")
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_ablation_config(args.config)
    runner = KNNSelectionAblationRunner(config)
    if args.command == "summarize":
        paths = runner.store.summarize()
        for name, path in paths.items():
            print(f"KNN_ABLATION_SUMMARY {name}={path}")
        return 0
    report = runner.run(
        targets=args.targets,
        seeds=args.seeds,
        max_tasks=args.max_tasks,
        force=args.force,
        fail_fast=args.fail_fast,
    )
    print(
        "KNN_ABLATION_COMPLETE "
        f"fingerprint={report.fingerprint} selected={report.selected} "
        f"attempted={report.attempted} succeeded={report.succeeded} "
        f"failed={report.failed} skipped={report.skipped} "
        f"output={report.output_root}",
        flush=True,
    )
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
