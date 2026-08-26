from __future__ import annotations

import os
import hashlib
import platform
import sys
import threading
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from cpdp.config import ExperimentConfig
from cpdp.data import load_all_projects, sources_for_target
from cpdp.methods.registry import (
    canonical_method_name,
    create_method,
    load_method_plugin,
)
from cpdp.metrics import METRIC_NAMES, evaluate_predictions
from cpdp.randomness import seed_everything
from cpdp.result_store import ResultStore
from cpdp.types import LabeledProject, PredictionResult
from threadpoolctl import threadpool_limits


SUPPORTED_PROTOCOL = "within-family-multisource-transductive-lopo"


def _effective_result_signature(
    project_root: Path, params: Mapping[str, Any]
) -> Optional[str]:
    configured = params.get("result_signature")
    reuse = params.get("oversampling_reuse", {})
    if not isinstance(reuse, Mapping) or not bool(reuse.get("enabled", False)):
        return str(configured) if configured is not None else None
    raw_path = Path(
        os.path.expandvars(
            os.path.expanduser(
                str(
                    reuse.get(
                        "seed_map",
                        "outputs/ipgdo_seed_search/best_oversampling_seeds.yaml",
                    )
                )
            )
        )
    )
    seed_map_path = raw_path if raw_path.is_absolute() else project_root / raw_path
    if not seed_map_path.is_file():
        raise FileNotFoundError(f"IPGDO best-seed map not found: {seed_map_path}")
    digest = hashlib.sha256(seed_map_path.read_bytes()).hexdigest()[:12]
    base = str(configured) if configured is not None else "ipgdo"
    return f"{base}-seedmap-{digest}"


def _method_uses_exclusive_cuda(
    config: ExperimentConfig, profile: str, method: str
) -> bool:
    params = config.method_params(profile, method)
    if method == "fedcpdp":
        if str(params.get("backend", "numpy")).lower() != "torch":
            return False
        requested_devices = (str(params.get("device", "auto")).lower(),)
    elif method in {
        "ipgdo_kan_jepa",
        "ipgdo_transformer_jepa",
    }:
        requested_devices = tuple(
            str(section.get("device", "auto")).lower()
            for section_name in ("oversampling", "predictor")
            for section in (params.get(section_name, {}),)
            if isinstance(section, Mapping)
        )
    else:
        return False

    if "cuda" in requested_devices:
        return True
    if "auto" not in requested_devices:
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


@dataclass(frozen=True)
class RunReport:
    selected: int
    attempted: int
    succeeded: int
    failed: int
    skipped: int


@dataclass(frozen=True)
class RunTask:
    method: str
    family: str
    target: str
    seed: int
    profile: str
    force: bool
    outer_workers: int = 1


@dataclass(frozen=True)
class TaskOutcome:
    succeeded: bool
    error_message: Optional[str] = None


def _sequential_task_groups(
    config: ExperimentConfig, tasks: Sequence[RunTask]
) -> Tuple[Tuple[RunTask, ...], ...]:
    groups = []
    index = 0
    while index < len(tasks):
        task = tasks[index]
        params = config.method_params(task.profile, task.method)
        configured_size = int(params.get("outer_seed_batch_size", 1))
        if configured_size < 1:
            raise ValueError("outer_seed_batch_size must be at least 1")
        can_batch = (
            configured_size > 1
            and task.method == "fedcpdp"
            and bool(params.get("batch_ensemble_on_device", True))
            and _method_uses_exclusive_cuda(config, task.profile, task.method)
        )
        if not can_batch:
            groups.append((task,))
            index += 1
            continue

        key = (task.method, task.family, task.target, task.profile, task.force)
        group = [task]
        cursor = index + 1
        while cursor < len(tasks) and len(group) < configured_size:
            candidate = tasks[cursor]
            candidate_key = (
                candidate.method,
                candidate.family,
                candidate.target,
                candidate.profile,
                candidate.force,
            )
            if candidate_key != key:
                break
            group.append(candidate)
            cursor += 1
        groups.append(tuple(group))
        index = cursor
    return tuple(groups)


_PROCESS_RUNNER: Optional["ExperimentRunner"] = None
_PROCESS_PROJECTS: Optional[Mapping[str, LabeledProject]] = None
_PROCESS_THREAD_LIMITER: Any = None


def _initialize_process_worker(
    config: ExperimentConfig,
    plugin_paths: Tuple[str, ...],
    blas_threads: int,
) -> None:
    global _PROCESS_RUNNER, _PROCESS_PROJECTS, _PROCESS_THREAD_LIMITER
    for plugin_path in plugin_paths:
        load_method_plugin(Path(plugin_path))
    _PROCESS_THREAD_LIMITER = threadpool_limits(limits=blas_threads)
    _PROCESS_RUNNER = ExperimentRunner(config)
    _PROCESS_PROJECTS = load_all_projects(config.data)


def _run_process_task(task: RunTask) -> TaskOutcome:
    if _PROCESS_RUNNER is None or _PROCESS_PROJECTS is None:
        raise RuntimeError("Experiment worker was not initialized")
    return _PROCESS_RUNNER._execute_task(task, _PROCESS_PROJECTS, verbose=True)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _package_versions() -> Dict[str, str]:
    versions = {}
    for package in (
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "imbalanced-learn",
        "torch",
    ):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            continue
    return versions


def validate_protocol(config: ExperimentConfig) -> None:
    if config.protocol != SUPPORTED_PROTOCOL:
        raise ValueError(
            f"Unsupported protocol {config.protocol!r}; expected {SUPPORTED_PROTOCOL!r}"
        )
    if config.target_labels != "evaluation-only":
        raise ValueError("Unified experiments require target_labels='evaluation-only'")
    unknown_metrics = sorted(set(config.metrics) - set(METRIC_NAMES))
    if unknown_metrics:
        raise ValueError(f"Unknown configured metrics: {', '.join(unknown_metrics)}")


def validate_datasets(
    config: ExperimentConfig,
) -> Tuple[Mapping[str, LabeledProject], Sequence[Dict[str, Any]]]:
    validate_protocol(config)
    projects = load_all_projects(config.data)
    records = []
    for family, project_name in config.targets():
        project = projects[f"{family}/{project_name}"]
        labels, counts = np.unique(project.labels, return_counts=True)
        class_counts = {int(label): int(count) for label, count in zip(labels, counts)}
        records.append(
            {
                "family": family,
                "project": project_name,
                "rows": project.features.n_samples,
                "features": project.features.n_features,
                "clean": class_counts.get(0, 0),
                "defective": class_counts.get(1, 0),
                "defect_rate": float(np.mean(project.labels)),
                "effort_column": project.features.effort_name,
                "cleaning": project.features.cleaning,
                "path": str(
                    config.data.root
                    / family
                    / f"{project_name}{config.data.file_suffix}"
                ),
            }
        )
    return projects, records


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig) -> None:
        validate_protocol(config)
        self.config = config
        self.project_root = config.source_path.parent.parent
        self.store = ResultStore(config.output_dir, config.save_predictions)
        self.environment = {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": _package_versions(),
        }

    def _targets(self, requested: Optional[Sequence[str]]) -> Tuple[Tuple[str, str], ...]:
        configured = tuple(self.config.targets())
        if not requested:
            return configured
        lookup = {f"{family}/{target}".lower(): (family, target) for family, target in configured}
        selected = []
        for value in requested:
            normalized = value.replace("\\", "/").strip().lower()
            if normalized not in lookup:
                raise KeyError(f"Unknown configured target: {value}")
            selected.append(lookup[normalized])
        return tuple(dict.fromkeys(selected))

    def _seeds(self, requested: Optional[Sequence[int]]) -> Tuple[int, ...]:
        seeds = tuple(int(seed) for seed in (requested or self.config.seeds))
        if not seeds:
            raise ValueError("At least one random seed is required")
        return tuple(dict.fromkeys(seeds))

    @staticmethod
    def _evaluation_arrays(
        target: LabeledProject, prediction: PredictionResult
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        scores = np.asarray(prediction.scores, dtype=np.float64).reshape(-1)
        if prediction.evaluated_row_ids is None:
            indices = np.arange(target.features.n_samples, dtype=np.int64)
        else:
            requested_ids = np.asarray(prediction.evaluated_row_ids).reshape(-1)
            if len(np.unique(requested_ids)) != len(requested_ids):
                raise ValueError("evaluated_row_ids contains duplicates")
            positions = {
                int(row_id): index
                for index, row_id in enumerate(target.features.row_ids.tolist())
            }
            missing = [int(row_id) for row_id in requested_ids if int(row_id) not in positions]
            if missing:
                raise ValueError(f"Prediction references unknown target row IDs: {missing[:5]}")
            indices = np.asarray([positions[int(row_id)] for row_id in requested_ids])
        if len(scores) != len(indices):
            raise ValueError(
                f"Method returned {len(scores)} scores for {len(indices)} evaluated rows"
            )
        if not np.isfinite(scores).all():
            raise ValueError("Method returned non-finite prediction scores")
        return (
            target.labels[indices],
            scores,
            target.features.effort[indices],
            target.features.row_ids[indices],
        )

    @staticmethod
    def _heartbeat(
        stop_event: threading.Event,
        task_name: str,
        started: float,
        interval_seconds: float,
    ) -> None:
        while not stop_event.wait(interval_seconds):
            elapsed = time.perf_counter() - started
            print(f"WORK {task_name} elapsed={elapsed:.0f}s", flush=True)

    def _execute_task(
        self,
        task: RunTask,
        projects: Mapping[str, LabeledProject],
        *,
        verbose: bool,
    ) -> TaskOutcome:
        method_name = task.method
        family = task.family
        target_name = task.target
        seed = task.seed
        started_at = _utc_now()
        started = time.perf_counter()
        task_name = f"{method_name} {family}/{target_name} seed={seed}"
        base_payload: Dict[str, Any] = {
            "schema_version": 1,
            "status": "running",
            "method": method_name,
            "family": family,
            "target": target_name,
            "seed": int(seed),
            "profile": task.profile,
            "protocol": self.config.protocol,
            "target_label_policy": self.config.target_labels,
            "config_file": str(self.config.source_path),
            "started_at": started_at,
        }
        if verbose:
            print(f"RUN  {task_name}", flush=True)

        heartbeat_stop = threading.Event()
        heartbeat_thread: Optional[threading.Thread] = None
        if verbose:
            heartbeat_thread = threading.Thread(
                target=self._heartbeat,
                args=(heartbeat_stop, task_name, started, 60.0),
                daemon=True,
            )
            heartbeat_thread.start()

        try:
            params = self.config.method_params(task.profile, method_name)
            effective_signature = _effective_result_signature(self.project_root, params)
            if effective_signature is not None:
                params["result_signature"] = effective_signature
            if task.outer_workers > 1 and "parallel_workers" in params:
                params["parallel_workers"] = 1
            method = create_method(method_name, self.project_root, params)
            target = projects[f"{family}/{target_name}"]
            sources = tuple(sources_for_target(projects, family, target_name))
            seed_everything(seed)
            prediction = method.fit_predict(sources, target.without_labels(), seed)
            labels, scores, effort, row_ids = self._evaluation_arrays(target, prediction)
            threshold = float(prediction.threshold)
            if not np.isfinite(threshold):
                raise ValueError("Method returned a non-finite classification threshold")
            all_metrics = evaluate_predictions(labels, scores, effort, threshold)
            payload = {
                **base_payload,
                "status": "success",
                "finished_at": _utc_now(),
                "duration_seconds": float(time.perf_counter() - started),
                "threshold": threshold,
                "metrics": {name: all_metrics[name] for name in self.config.metrics},
                "source_projects": [source.key for source in sources],
                "source_rows": {source.key: source.features.n_samples for source in sources},
                "target_rows": int(len(labels)),
                "target_cleaning": target.features.cleaning,
                "method_params": params,
                "method_metadata": prediction.metadata,
                "provenance": asdict(method.provenance),
                "environment": self.environment,
                "execution": {
                    "outer_workers": int(task.outer_workers),
                    "logical_cpus": int(os.cpu_count() or 1),
                },
            }
            self.store.save_success(_jsonable(payload), labels, scores, effort, row_ids)
            if verbose:
                metric_text = " ".join(
                    f"{name}={all_metrics[name]:.4f}" for name in self.config.metrics
                )
                print(
                    f"OK   {method_name} {family}/{target_name} {metric_text}",
                    flush=True,
                )
            return TaskOutcome(succeeded=True)
        except Exception as exc:
            payload = {
                **base_payload,
                "status": "failure",
                "finished_at": _utc_now(),
                "duration_seconds": float(time.perf_counter() - started),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
                "environment": self.environment,
                "execution": {
                    "outer_workers": int(task.outer_workers),
                    "logical_cpus": int(os.cpu_count() or 1),
                },
            }
            self.store.save_failure(_jsonable(payload))
            message = f"{type(exc).__name__}: {exc}"
            if verbose:
                print(f"FAIL {method_name} {family}/{target_name}: {message}", flush=True)
            return TaskOutcome(succeeded=False, error_message=message)
        finally:
            heartbeat_stop.set()
            if heartbeat_thread is not None:
                heartbeat_thread.join(timeout=1.0)

    def _execute_seed_batch(
        self,
        tasks: Sequence[RunTask],
        projects: Mapping[str, LabeledProject],
        *,
        verbose: bool,
    ) -> Tuple[TaskOutcome, ...]:
        if len(tasks) < 2:
            raise ValueError("A seed batch requires at least two tasks")
        first = tasks[0]
        key = (first.method, first.family, first.target, first.profile)
        if any(
            (task.method, task.family, task.target, task.profile) != key
            for task in tasks
        ):
            raise ValueError("A seed batch must share method, target, and profile")

        method_name, family, target_name, profile = key
        seeds = tuple(task.seed for task in tasks)
        started_at = _utc_now()
        started = time.perf_counter()
        task_name = (
            f"{method_name} {family}/{target_name} "
            f"seed_batch={seeds[0]}..{seeds[-1]} size={len(seeds)}"
        )
        if verbose:
            print(f"RUN  {task_name}", flush=True)

        heartbeat_stop = threading.Event()
        heartbeat_thread: Optional[threading.Thread] = None
        if verbose:
            heartbeat_thread = threading.Thread(
                target=self._heartbeat,
                args=(heartbeat_stop, task_name, started, 60.0),
                daemon=True,
            )
            heartbeat_thread.start()

        def base_payload(task: RunTask) -> Dict[str, Any]:
            return {
                "schema_version": 1,
                "status": "running",
                "method": method_name,
                "family": family,
                "target": target_name,
                "seed": int(task.seed),
                "profile": profile,
                "protocol": self.config.protocol,
                "target_label_policy": self.config.target_labels,
                "config_file": str(self.config.source_path),
                "started_at": started_at,
            }

        try:
            params = self.config.method_params(profile, method_name)
            method = create_method(method_name, self.project_root, params)
            fit_predict_many = getattr(method, "fit_predict_many", None)
            if not callable(fit_predict_many):
                raise TypeError(f"{method_name} does not support seed batching")
            target = projects[f"{family}/{target_name}"]
            sources = tuple(sources_for_target(projects, family, target_name))
            seed_everything(seeds[0])
            predictions = fit_predict_many(
                sources, target.without_labels(), seeds
            )
            if set(predictions) != set(seeds):
                raise ValueError(
                    "Batched method did not return exactly one prediction per seed"
                )

            prepared = []
            for task in tasks:
                prediction = predictions[task.seed]
                labels, scores, effort, row_ids = self._evaluation_arrays(
                    target, prediction
                )
                threshold = float(prediction.threshold)
                if not np.isfinite(threshold):
                    raise ValueError(
                        "Method returned a non-finite classification threshold"
                    )
                all_metrics = evaluate_predictions(
                    labels, scores, effort, threshold
                )
                prepared.append(
                    (
                        task,
                        prediction,
                        labels,
                        scores,
                        effort,
                        row_ids,
                        threshold,
                        all_metrics,
                    )
                )

            batch_duration = float(time.perf_counter() - started)
            amortized_duration = batch_duration / len(tasks)
            outcomes = []
            for (
                task,
                prediction,
                labels,
                scores,
                effort,
                row_ids,
                threshold,
                all_metrics,
            ) in prepared:
                payload = {
                    **base_payload(task),
                    "status": "success",
                    "finished_at": _utc_now(),
                    "duration_seconds": amortized_duration,
                    "threshold": threshold,
                    "metrics": {
                        name: all_metrics[name] for name in self.config.metrics
                    },
                    "source_projects": [source.key for source in sources],
                    "source_rows": {
                        source.key: source.features.n_samples for source in sources
                    },
                    "target_rows": int(len(labels)),
                    "target_cleaning": target.features.cleaning,
                    "method_params": params,
                    "method_metadata": prediction.metadata,
                    "provenance": asdict(method.provenance),
                    "environment": self.environment,
                    "execution": {
                        "outer_workers": 1,
                        "logical_cpus": int(os.cpu_count() or 1),
                        "seed_batch_size": len(tasks),
                        "seed_batch_wall_seconds": batch_duration,
                        "duration_accounting": "batch-wall-time-divided-by-seeds",
                    },
                }
                self.store.save_success(
                    _jsonable(payload), labels, scores, effort, row_ids
                )
                if verbose:
                    metric_text = " ".join(
                        f"{name}={all_metrics[name]:.4f}"
                        for name in self.config.metrics
                    )
                    print(
                        f"OK   {method_name} {family}/{target_name} "
                        f"seed={task.seed} {metric_text}",
                        flush=True,
                    )
                outcomes.append(TaskOutcome(succeeded=True))
            return tuple(outcomes)
        except Exception as exc:
            batch_duration = float(time.perf_counter() - started)
            message = f"{type(exc).__name__}: {exc}"
            for task in tasks:
                payload = {
                    **base_payload(task),
                    "status": "failure",
                    "finished_at": _utc_now(),
                    "duration_seconds": batch_duration,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                    "environment": self.environment,
                    "execution": {
                        "outer_workers": 1,
                        "logical_cpus": int(os.cpu_count() or 1),
                        "seed_batch_size": len(tasks),
                        "seed_batch_wall_seconds": batch_duration,
                    },
                }
                self.store.save_failure(_jsonable(payload))
            if verbose:
                print(
                    f"FAIL {method_name} {family}/{target_name} batch: {message}",
                    flush=True,
                )
            return tuple(
                TaskOutcome(succeeded=False, error_message=message)
                for _ in tasks
            )
        finally:
            heartbeat_stop.set()
            if heartbeat_thread is not None:
                heartbeat_thread.join(timeout=1.0)

    def run(
        self,
        methods: Sequence[str],
        *,
        profile: str,
        targets: Optional[Sequence[str]] = None,
        seeds: Optional[Sequence[int]] = None,
        max_runs: Optional[int] = None,
        workers: int = 1,
        force: bool = False,
        fail_fast: bool = False,
        plugin_paths: Sequence[Path] = (),
        verbose: bool = True,
    ) -> RunReport:
        selected_methods = tuple(
            dict.fromkeys(canonical_method_name(name) for name in methods)
        )
        selected_targets = self._targets(targets)
        selected_seeds = self._seeds(seeds)
        combinations = tuple(
            (method, family, target, seed)
            for method in selected_methods
            for family, target in selected_targets
            for seed in selected_seeds
        )
        if max_runs is not None and max_runs < 1:
            raise ValueError("max_runs must be at least 1")
        if workers < 1:
            raise ValueError("workers must be at least 1")
        if fail_fast and workers > 1:
            raise ValueError("fail_fast requires workers=1")

        pending_tasks = []
        skipped = 0
        for method_name, family, target_name, seed in combinations:
            configured_params = self.config.method_params(profile, method_name)
            result_signature = _effective_result_signature(
                self.project_root, configured_params
            )
            if not force and self.store.is_complete(
                profile,
                method_name,
                family,
                target_name,
                seed,
                result_signature=(
                    str(result_signature) if result_signature is not None else None
                ),
            ):
                skipped += 1
                if verbose:
                    print(f"SKIP {method_name} {family}/{target_name} seed={seed}")
                continue
            if max_runs is not None and len(pending_tasks) >= max_runs:
                break
            pending_tasks.append(
                RunTask(
                    method=method_name,
                    family=family,
                    target=target_name,
                    seed=int(seed),
                    profile=profile,
                    force=force,
                )
            )

        attempted = len(pending_tasks)
        succeeded = failed = 0
        if pending_tasks:
            effective_workers = min(int(workers), len(pending_tasks))
            exclusive_cuda_methods = sorted(
                {
                    task.method
                    for task in pending_tasks
                    if _method_uses_exclusive_cuda(
                        self.config, task.profile, task.method
                    )
                }
            )
            if exclusive_cuda_methods and effective_workers > 1:
                effective_workers = 1
                if verbose:
                    print(
                        "PARALLEL adjusted outer_workers=1 for exclusive CUDA method(s): "
                        + ", ".join(exclusive_cuda_methods),
                        flush=True,
                    )
            tasks = tuple(
                RunTask(
                    method=task.method,
                    family=task.family,
                    target=task.target,
                    seed=task.seed,
                    profile=task.profile,
                    force=task.force,
                    outer_workers=effective_workers,
                )
                for task in pending_tasks
            )
            if effective_workers == 1:
                projects, _ = validate_datasets(self.config)
                for task_group in _sequential_task_groups(self.config, tasks):
                    if len(task_group) == 1:
                        outcomes = (
                            self._execute_task(
                                task_group[0], projects, verbose=verbose
                            ),
                        )
                    else:
                        outcomes = self._execute_seed_batch(
                            task_group, projects, verbose=verbose
                        )
                    for outcome in outcomes:
                        succeeded += int(outcome.succeeded)
                        failed += int(not outcome.succeeded)
                        if fail_fast and not outcome.succeeded:
                            raise RuntimeError(
                                outcome.error_message or "Experiment task failed"
                            )
            else:
                validate_datasets(self.config)
                blas_threads = max(1, int(os.cpu_count() or 1) // effective_workers)
                if verbose:
                    print(
                        f"PARALLEL outer_workers={effective_workers} "
                        f"blas_threads_per_worker={blas_threads}",
                        flush=True,
                    )
                with ProcessPoolExecutor(
                    max_workers=effective_workers,
                    initializer=_initialize_process_worker,
                    initargs=(
                        self.config,
                        tuple(str(path.resolve()) for path in plugin_paths),
                        blas_threads,
                    ),
                ) as executor:
                    futures = [executor.submit(_run_process_task, task) for task in tasks]
                    for future in as_completed(futures):
                        try:
                            outcome = future.result()
                        except Exception as exc:
                            failed += 1
                            if verbose:
                                print(
                                    f"FAIL worker process: {type(exc).__name__}: {exc}",
                                    flush=True,
                                )
                        else:
                            succeeded += int(outcome.succeeded)
                            failed += int(not outcome.succeeded)

        return RunReport(
            selected=len(combinations),
            attempted=attempted,
            succeeded=succeeded,
            failed=failed,
            skipped=skipped,
        )
