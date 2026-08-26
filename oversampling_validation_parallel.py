from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Iterable, Optional, Sequence, Tuple, Union


WORKERS_ENV = "CPDP_OVERSAMPLING_VALIDATION_WORKERS"
ELASTIC_ENABLED_ENV = "CPDP_OVERSAMPLING_ELASTIC_GPU"
GPU_UTILIZATION_LIMIT_ENV = "CPDP_GPU_LAUNCH_UTILIZATION_LIMIT"
GPU_MIN_FREE_MEMORY_ENV = "CPDP_GPU_MIN_FREE_MEMORY_MB"
GPU_MAX_TEMPERATURE_ENV = "CPDP_GPU_MAX_TEMPERATURE_C"
GPU_POLL_SECONDS_ENV = "CPDP_GPU_POLL_SECONDS"
GPU_RAMP_UP_SECONDS_ENV = "CPDP_GPU_RAMP_UP_SECONDS"
GPU_PAUSE_PROCESSES_ENV = "CPDP_GPU_PAUSE_PROCESSES"
GPU_MAX_CONCURRENT_TASKS_ENV = "CPDP_GPU_MAX_CONCURRENT_TASKS"
GPU_TRANSIENT_RETRIES_ENV = "CPDP_GPU_TRANSIENT_RETRIES"
GPU_RETRY_COOLDOWN_SECONDS_ENV = "CPDP_GPU_RETRY_COOLDOWN_SECONDS"

TRANSIENT_CUDA_FAILURE_MARKERS = (
    "cuda error: unspecified launch failure",
    "cuda error: an illegal memory access was encountered",
    "cuda error: launch timeout",
    "cuda error: the launch timed out",
    "cuda error: unknown error",
    "cuda initialization error",
    "cuda driver error",
    "cudnn_status_internal_error",
)


@dataclass(frozen=True)
class ParallelControl:
    argv: Tuple[str, ...]
    workers: int
    worker_process: bool


@dataclass(frozen=True)
class RunSelection:
    config_path: Path
    profile: str
    protocols: Tuple[str, ...]
    variants: Tuple[str, ...]
    families: Tuple[str, ...]
    targets: Tuple[str, ...]
    seeds: Tuple[int, ...]
    folds: Tuple[int, ...]
    max_tasks: Optional[int]
    fail_fast: bool
    force: bool
    fingerprint: str
    result_root: Path


@dataclass(frozen=True)
class WorkerSpec:
    index: int
    label: str
    argv: Tuple[str, ...]
    result_path: Optional[Path] = None


@dataclass(frozen=True)
class ParallelPhase:
    protocol: str
    workers: Tuple[WorkerSpec, ...]


@dataclass(frozen=True)
class GpuSnapshot:
    utilization_percent: float
    memory_used_mb: float
    memory_total_mb: float
    temperature_c: float

    @property
    def free_memory_mb(self) -> float:
        return self.memory_total_mb - self.memory_used_mb


@dataclass(frozen=True)
class ElasticGpuConfig:
    enabled: bool
    max_workers: int
    max_concurrent_tasks: int
    launch_utilization_limit: float
    minimum_free_memory_mb: float
    maximum_temperature_c: float
    poll_seconds: float
    ramp_up_seconds: float
    pause_process_names: Tuple[str, ...]
    transient_cuda_retries: int
    retry_cooldown_seconds: float

    def validate(self) -> None:
        if self.max_workers < 1:
            raise ValueError("elastic GPU max_workers must be at least 1")
        if self.max_concurrent_tasks < 1:
            raise ValueError("GPU max concurrent tasks must be at least 1")
        if not 0.0 < self.launch_utilization_limit <= 100.0:
            raise ValueError("GPU utilization limit must be in (0, 100]")
        if self.minimum_free_memory_mb < 0.0:
            raise ValueError("GPU minimum free memory cannot be negative")
        if self.maximum_temperature_c <= 0.0:
            raise ValueError("GPU maximum temperature must be positive")
        if self.poll_seconds <= 0.0 or self.ramp_up_seconds < 0.0:
            raise ValueError("GPU polling values must be nonnegative")
        if self.transient_cuda_retries < 0:
            raise ValueError("CUDA transient retries cannot be negative")
        if self.retry_cooldown_seconds < 0.0:
            raise ValueError("CUDA retry cooldown cannot be negative")


def parse_parallel_control(
    argv: Sequence[str], default_workers: int
) -> ParallelControl:
    workers = int(default_workers)
    worker_process = False
    remaining = []
    index = 0
    while index < len(argv):
        argument = str(argv[index])
        if argument == "--parallel-worker":
            worker_process = True
            index += 1
            continue
        if argument == "--workers":
            if index + 1 >= len(argv):
                raise ValueError("--workers requires an integer value")
            workers = int(argv[index + 1])
            index += 2
            continue
        if argument.startswith("--workers="):
            workers = int(argument.split("=", 1)[1])
            index += 1
            continue
        remaining.append(argument)
        index += 1
    if workers < 1:
        raise ValueError("--workers must be at least 1")
    return ParallelControl(tuple(remaining), workers, worker_process)


def configured_default_workers(fallback: int = 4) -> int:
    raw_value = os.environ.get(WORKERS_ENV, str(fallback))
    try:
        workers = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{WORKERS_ENV} must be an integer") from exc
    if workers < 1:
        raise ValueError(f"{WORKERS_ENV} must be at least 1")
    return workers


def _environment_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def _environment_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return float(default)
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric") from exc


def _environment_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return int(default)
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _normalized_process_name(value: str) -> str:
    return Path(str(value).strip()).name.lower()


def configured_elastic_gpu(max_workers: int) -> ElasticGpuConfig:
    process_names = tuple(
        dict.fromkeys(
            _normalized_process_name(value)
            for value in os.environ.get(GPU_PAUSE_PROCESSES_ENV, "")
            .replace(",", ";")
            .split(";")
            if value.strip()
        )
    )
    config = ElasticGpuConfig(
        enabled=_environment_bool(ELASTIC_ENABLED_ENV, True),
        max_workers=int(max_workers),
        max_concurrent_tasks=_environment_int(
            GPU_MAX_CONCURRENT_TASKS_ENV, min(int(max_workers), 2)
        ),
        launch_utilization_limit=_environment_float(
            GPU_UTILIZATION_LIMIT_ENV, 75.0
        ),
        minimum_free_memory_mb=_environment_float(
            GPU_MIN_FREE_MEMORY_ENV, 2048.0
        ),
        maximum_temperature_c=_environment_float(
            GPU_MAX_TEMPERATURE_ENV, 80.0
        ),
        poll_seconds=_environment_float(GPU_POLL_SECONDS_ENV, 2.0),
        ramp_up_seconds=_environment_float(GPU_RAMP_UP_SECONDS_ENV, 8.0),
        pause_process_names=process_names,
        transient_cuda_retries=_environment_int(GPU_TRANSIENT_RETRIES_ENV, 3),
        retry_cooldown_seconds=_environment_float(
            GPU_RETRY_COOLDOWN_SECONDS_ENV, 30.0
        ),
    )
    config.validate()
    return config


def parse_run_selection(
    argv: Sequence[str], project_root: Path
) -> Optional[RunSelection]:
    if "run" not in argv or "--help" in argv or "-h" in argv:
        return None
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--config",
        default=str(project_root / "configs" / "oversampling_validation.yaml"),
    )
    commands = parser.add_subparsers(dest="command")
    run_parser = commands.add_parser("run", add_help=False)
    run_parser.add_argument("--profile", default="smoke")
    run_parser.add_argument("--protocol", action="append")
    run_parser.add_argument("--variant", action="append")
    run_parser.add_argument("--family", action="append")
    run_parser.add_argument("--target", action="append")
    run_parser.add_argument("--seed", action="append", type=int)
    run_parser.add_argument("--fold", action="append", type=int)
    run_parser.add_argument("--max-tasks", type=int)
    run_parser.add_argument("--force", action="store_true")
    run_parser.add_argument("--fail-fast", action="store_true")
    try:
        arguments, _ = parser.parse_known_args(list(argv))
    except SystemExit:
        return None
    if arguments.command != "run":
        return None

    from cpdp.oversampling_validation.config import load_validation_config

    config_path = Path(arguments.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    config_path = config_path.resolve()
    config = load_validation_config(config_path)
    profile = config.profile(arguments.profile)

    if not arguments.variant or tuple(arguments.variant) == ("all",):
        variants = profile.variants
    else:
        variants = tuple(dict.fromkeys(str(value) for value in arguments.variant))
        unknown_variants = sorted(set(variants) - set(profile.variants))
        if unknown_variants:
            raise KeyError(
                "Variants are not enabled in profile "
                f"{arguments.profile}: {', '.join(unknown_variants)}"
            )

    configured_families = tuple(
        family.name for family in config.experiment.data.families
    )
    requested_families = tuple(arguments.family or configured_families)
    unknown_families = sorted(set(requested_families) - set(configured_families))
    if unknown_families:
        raise KeyError(f"Unknown dataset families: {', '.join(unknown_families)}")
    families = tuple(dict.fromkeys(requested_families))

    configured_targets = tuple(
        f"{family}/{target}" for family, target in config.experiment.targets()
    )
    target_lookup = {target.lower(): target for target in configured_targets}
    if arguments.target:
        selected_targets = []
        for requested in arguments.target:
            normalized = str(requested).replace("\\", "/").strip().lower()
            if normalized not in target_lookup:
                raise KeyError(f"Unknown target project: {requested}")
            selected_targets.append(target_lookup[normalized])
        targets = tuple(dict.fromkeys(selected_targets))
    else:
        targets = tuple(
            target
            for target in configured_targets
            if target.split("/", 1)[0] in families
        )
    invalid_targets = [
        target for target in targets if target.split("/", 1)[0] not in families
    ]
    if invalid_targets:
        raise ValueError(
            "Target projects must belong to selected families: "
            + ", ".join(invalid_targets)
        )

    default_protocols = tuple(
        protocol
        for protocol, enabled in (
            ("source_cv", profile.source_cv.enabled),
            ("cpdp", profile.cpdp.enabled),
        )
        if enabled
    )
    protocols = tuple(dict.fromkeys(arguments.protocol or default_protocols))
    seeds = tuple(dict.fromkeys(int(seed) for seed in (arguments.seed or profile.seeds)))
    if not seeds:
        raise ValueError("At least one random seed is required")
    configured_folds = tuple(range(profile.source_cv.folds))
    folds = tuple(
        dict.fromkeys(int(fold) for fold in (arguments.fold or configured_folds))
    )
    unknown_folds = sorted(set(folds) - set(configured_folds))
    if unknown_folds:
        raise ValueError(
            f"Folds are not enabled in profile {arguments.profile}: {unknown_folds}"
        )

    from cpdp.oversampling_validation.runner import OversamplingValidationRunner

    fingerprint = OversamplingValidationRunner(config).fingerprint(arguments.profile)
    return RunSelection(
        config_path=config_path,
        profile=arguments.profile,
        protocols=protocols,
        variants=variants,
        families=families,
        targets=targets,
        seeds=seeds,
        folds=folds,
        max_tasks=arguments.max_tasks,
        fail_fast=bool(arguments.fail_fast),
        force=bool(arguments.force),
        fingerprint=fingerprint,
        result_root=config.output_dir / fingerprint / arguments.profile,
    )


def _remove_options(argv: Sequence[str], option_names: Iterable[str]) -> list[str]:
    names = set(option_names)
    remaining = []
    index = 0
    while index < len(argv):
        argument = str(argv[index])
        matched = next(
            (
                name
                for name in names
                if argument == name or argument.startswith(f"{name}=")
            ),
            None,
        )
        if matched is None:
            remaining.append(argument)
            index += 1
            continue
        index += 2 if argument == matched else 1
    return remaining


def _round_robin(
    values: Sequence[Union[str, int]], workers: int
) -> Tuple[Tuple[Union[str, int], ...], ...]:
    shard_count = min(max(1, int(workers)), len(values))
    shards = [[] for _ in range(shard_count)]
    for index, value in enumerate(values):
        shards[index % shard_count].append(value)
    return tuple(tuple(shard) for shard in shards if shard)


def _append_values(command: list[str], option: str, values: Sequence[object]) -> None:
    for value in values:
        command.extend((option, str(value)))


def _phase_worker_arguments(
    base_argv: Sequence[str],
    protocol: str,
    *,
    seeds: Sequence[int],
    variants: Sequence[str] = (),
    families: Sequence[str] = (),
    targets: Sequence[str] = (),
    folds: Sequence[int] = (),
) -> Tuple[str, ...]:
    command = _remove_options(
        base_argv,
        ("--protocol", "--variant", "--family", "--target", "--seed", "--fold"),
    )
    command.extend(("--protocol", protocol))
    _append_values(command, "--variant", variants)
    _append_values(command, "--family", families)
    _append_values(command, "--target", targets)
    _append_values(command, "--seed", seeds)
    _append_values(command, "--fold", folds)
    return tuple(command)


def _atomic_result_path(
    selection: RunSelection,
    protocol: str,
    variant: str,
    family: str,
    seed: int,
    *,
    fold: Optional[int] = None,
    target: Optional[str] = None,
) -> Path:
    base = selection.result_root / "runs" / protocol / variant / family
    if protocol == "source_cv":
        if fold is None:
            raise ValueError("source_cv task requires a fold")
        return base / f"seed_{seed}" / f"fold_{fold}.json"
    if protocol == "cpdp":
        if target is None:
            raise ValueError("cpdp task requires a target")
        return base / target / f"seed_{seed}.json"
    raise ValueError(f"Unknown validation protocol: {protocol}")


def _build_atomic_phase(
    argv: Sequence[str], selection: RunSelection, protocol: str
) -> ParallelPhase:
    specifications = []
    if protocol == "source_cv":
        for variant in selection.variants:
            for family in selection.families:
                for seed in selection.seeds:
                    for fold in selection.folds:
                        specifications.append(
                            WorkerSpec(
                                index=len(specifications) + 1,
                                label=(
                                    f"source_cv:{variant}:{family}:"
                                    f"seed={seed}:fold={fold}"
                                ),
                                argv=_phase_worker_arguments(
                                    argv,
                                    protocol,
                                    variants=(variant,),
                                    families=(family,),
                                    seeds=(seed,),
                                    folds=(fold,),
                                ),
                                result_path=_atomic_result_path(
                                    selection,
                                    protocol,
                                    variant,
                                    family,
                                    seed,
                                    fold=fold,
                                ),
                            )
                        )
    elif protocol == "cpdp":
        for variant in selection.variants:
            for target_key in selection.targets:
                family, target = target_key.split("/", 1)
                for seed in selection.seeds:
                    specifications.append(
                        WorkerSpec(
                            index=len(specifications) + 1,
                            label=f"cpdp:{variant}:{target_key}:seed={seed}",
                            argv=_phase_worker_arguments(
                                argv,
                                protocol,
                                variants=(variant,),
                                families=(family,),
                                targets=(target_key,),
                                seeds=(seed,),
                            ),
                            result_path=_atomic_result_path(
                                selection,
                                protocol,
                                variant,
                                family,
                                seed,
                                target=target,
                            ),
                        )
                    )
    else:
        raise ValueError(f"Unknown validation protocol: {protocol}")
    return ParallelPhase(protocol, tuple(specifications))


def build_parallel_phases(
    argv: Sequence[str], project_root: Path, workers: int, *, atomic: bool = False
) -> Tuple[ParallelPhase, ...]:
    selection = parse_run_selection(argv, project_root)
    if selection is None or selection.max_tasks is not None:
        return ()

    if atomic:
        return tuple(
            _build_atomic_phase(argv, selection, protocol)
            for protocol in selection.protocols
        )

    phases = []
    for protocol in selection.protocols:
        specifications = []
        if protocol == "cpdp":
            primary_values: Sequence[str] = selection.targets
            primary_option = "target"
        elif protocol == "source_cv":
            primary_values = selection.families
            primary_option = "family"
        else:
            raise ValueError(f"Unknown validation protocol: {protocol}")

        if len(primary_values) >= workers or len(selection.seeds) < len(primary_values):
            primary_shards = _round_robin(primary_values, workers)
            for index, shard in enumerate(primary_shards, start=1):
                string_shard = tuple(str(value) for value in shard)
                worker_argv = _phase_worker_arguments(
                    argv,
                    protocol,
                    variants=selection.variants,
                    families=string_shard if primary_option == "family" else (),
                    targets=string_shard if primary_option == "target" else (),
                    seeds=selection.seeds,
                    folds=selection.folds if protocol == "source_cv" else (),
                )
                specifications.append(
                    WorkerSpec(
                        index=index,
                        label=f"{protocol}:{primary_option}s={len(shard)}",
                        argv=worker_argv,
                    )
                )
        else:
            seed_shards = _round_robin(selection.seeds, workers)
            for index, shard in enumerate(seed_shards, start=1):
                worker_argv = _phase_worker_arguments(
                    argv,
                    protocol,
                    variants=selection.variants,
                    families=selection.families if primary_option == "family" else (),
                    targets=selection.targets if primary_option == "target" else (),
                    seeds=tuple(int(seed) for seed in shard),
                    folds=selection.folds if protocol == "source_cv" else (),
                )
                specifications.append(
                    WorkerSpec(
                        index=index,
                        label=f"{protocol}:seeds={len(shard)}",
                        argv=worker_argv,
                    )
                )
        phases.append(ParallelPhase(protocol, tuple(specifications)))
    return tuple(phases)


def _stream_output(
    process: subprocess.Popen[str], prefix: str, output_lock: threading.Lock
) -> None:
    if process.stdout is None:
        return
    for line in process.stdout:
        with output_lock:
            print(f"{prefix} {line}", end="", flush=True)


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            capture_output=True,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return


def query_nvidia_gpu() -> Optional[GpuSnapshot]:
    command = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5.0,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    first_line = next(
        (line.strip() for line in completed.stdout.splitlines() if line.strip()),
        "",
    )
    if not first_line:
        return None
    try:
        values = [float(value.strip()) for value in first_line.split(",")]
    except ValueError:
        return None
    if len(values) != 4:
        return None
    return GpuSnapshot(
        utilization_percent=values[0],
        memory_used_mb=values[1],
        memory_total_mb=values[2],
        temperature_c=values[3],
    )


def query_running_process_names() -> Tuple[str, ...]:
    try:
        if os.name == "nt":
            completed = subprocess.run(
                ["tasklist", "/FO", "CSV", "/NH"],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5.0,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            if completed.returncode != 0:
                return ()
            rows = csv.reader(completed.stdout.splitlines())
            names = (
                _normalized_process_name(row[0])
                for row in rows
                if row and row[0].strip()
            )
        else:
            completed = subprocess.run(
                ["ps", "-eo", "comm="],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5.0,
            )
            if completed.returncode != 0:
                return ()
            names = (
                _normalized_process_name(line)
                for line in completed.stdout.splitlines()
                if line.strip()
            )
        return tuple(dict.fromkeys(names))
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return ()


def elastic_block_reason(
    snapshot: Optional[GpuSnapshot],
    running_process_names: Sequence[str],
    config: ElasticGpuConfig,
) -> Optional[str]:
    running = {
        _normalized_process_name(name)
        for name in running_process_names
        if str(name).strip()
    }
    matched = tuple(
        name for name in config.pause_process_names if name in running
    )
    if matched:
        return "pause process active: " + ",".join(matched)
    if snapshot is None:
        return None
    if snapshot.temperature_c >= config.maximum_temperature_c:
        return (
            f"GPU temperature {snapshot.temperature_c:.0f}C >= "
            f"{config.maximum_temperature_c:.0f}C"
        )
    if snapshot.utilization_percent >= config.launch_utilization_limit:
        return (
            f"GPU utilization {snapshot.utilization_percent:.0f}% >= "
            f"{config.launch_utilization_limit:.0f}%"
        )
    if snapshot.free_memory_mb < config.minimum_free_memory_mb:
        return (
            f"GPU free memory {snapshot.free_memory_mb:.0f} MiB < "
            f"{config.minimum_free_memory_mb:.0f} MiB"
        )
    return None


def _is_successful_result(path: Path, fingerprint: str) -> bool:
    if not path.is_file():
        return False
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("status") == "success"
        and payload.get("fingerprint") == fingerprint
    )


def _failure_text(path: Optional[Path]) -> str:
    if path is None or not path.is_file():
        return ""
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return ""
    if payload.get("status") != "failure":
        return ""
    return "\n".join(
        str(payload.get(name, ""))
        for name in ("error_type", "error_message", "traceback")
    )


def _is_transient_cuda_failure(path: Optional[Path]) -> bool:
    failure = _failure_text(path).lower()
    return any(marker in failure for marker in TRANSIENT_CUDA_FAILURE_MARKERS)


def _launch_worker(
    specification: WorkerSpec,
    *,
    python_executable: Path,
    runner_script: Path,
    project_root: Path,
    environment: dict[str, str],
    creation_flags: int,
    output_lock: threading.Lock,
) -> Tuple[subprocess.Popen[str], threading.Thread]:
    command = [
        str(python_executable),
        str(runner_script),
        "--parallel-worker",
        *specification.argv,
    ]
    process = subprocess.Popen(
        command,
        cwd=project_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=environment,
        creationflags=creation_flags,
        start_new_session=os.name != "nt",
    )
    thread = threading.Thread(
        target=_stream_output,
        args=(process, f"[T{specification.index}]", output_lock),
        daemon=True,
    )
    thread.start()
    return process, thread


def _run_elastic_phase(
    phase: ParallelPhase,
    *,
    selection: RunSelection,
    config: ElasticGpuConfig,
    python_executable: Path,
    runner_script: Path,
    project_root: Path,
) -> int:
    skipped = 0
    pending: Deque[WorkerSpec] = deque()
    for specification in phase.workers:
        if (
            not selection.force
            and specification.result_path is not None
            and _is_successful_result(
                specification.result_path, selection.fingerprint
            )
        ):
            skipped += 1
        else:
            pending.append(specification)

    print(
        f"ELASTIC phase={phase.protocol} selected={len(phase.workers)} "
        f"pending={len(pending)} skipped={skipped} "
        f"max_workers={config.max_workers} "
        f"gpu_task_limit={min(config.max_workers, config.max_concurrent_tasks)}",
        flush=True,
    )
    if not pending:
        return 0

    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    output_lock = threading.Lock()
    active: dict[int, Tuple[WorkerSpec, subprocess.Popen[str], threading.Thread]] = {}
    succeeded = 0
    failed = 0
    retried = 0
    retry_counts: dict[int, int] = {}
    adaptive_worker_limit = min(config.max_workers, config.max_concurrent_tasks)
    next_launch_time = 0.0
    last_status: Optional[str] = None
    last_status_time = 0.0
    telemetry_warning_shown = False

    try:
        while pending or active:
            for process_id, (specification, process, thread) in tuple(active.items()):
                return_code = process.poll()
                if return_code is None:
                    continue
                thread.join(timeout=5.0)
                del active[process_id]
                result_succeeded = (
                    specification.result_path is None
                    or _is_successful_result(
                        specification.result_path, selection.fingerprint
                    )
                )
                if return_code == 0 and result_succeeded:
                    succeeded += 1
                    outcome = "success"
                elif (
                    _is_transient_cuda_failure(specification.result_path)
                    and retry_counts.get(specification.index, 0)
                    < config.transient_cuda_retries
                ):
                    retry_counts[specification.index] = (
                        retry_counts.get(specification.index, 0) + 1
                    )
                    retry_attempt = retry_counts[specification.index]
                    retried += 1
                    pending.appendleft(specification)
                    adaptive_worker_limit = max(1, adaptive_worker_limit - 1)
                    next_launch_time = max(
                        next_launch_time,
                        time.monotonic() + config.retry_cooldown_seconds,
                    )
                    outcome = (
                        f"cuda-retry({retry_attempt}/"
                        f"{config.transient_cuda_retries})"
                    )
                    print(
                        f"ELASTIC CUDA recovery task={specification.index} "
                        f"cooldown={config.retry_cooldown_seconds:.0f}s "
                        f"gpu_task_limit={adaptive_worker_limit}",
                        flush=True,
                    )
                else:
                    failed += 1
                    outcome = f"failed({return_code})"
                print(
                    f"ELASTIC complete task={specification.index} "
                    f"outcome={outcome} active={len(active)} "
                    f"pending={len(pending)}",
                    flush=True,
                )
                if outcome.startswith("failed") and selection.fail_fast:
                    for _, other, _ in active.values():
                        _terminate_process_tree(other)
                    pending.clear()
                    break

            if selection.fail_fast and failed:
                break
            if not pending:
                if active:
                    time.sleep(config.poll_seconds)
                continue

            now = time.monotonic()
            snapshot = query_nvidia_gpu()
            running_names = (
                query_running_process_names() if config.pause_process_names else ()
            )
            effective_max_workers = adaptive_worker_limit
            if snapshot is None:
                effective_max_workers = 1
                if not telemetry_warning_shown:
                    print(
                        "ELASTIC warning: NVIDIA telemetry unavailable; "
                        "falling back to one active task.",
                        flush=True,
                    )
                    telemetry_warning_shown = True

            block_reason = elastic_block_reason(snapshot, running_names, config)
            if len(active) >= effective_max_workers:
                block_reason = (
                    f"active task limit reached: "
                    f"{len(active)}/{effective_max_workers}"
                )
            elif now < next_launch_time:
                block_reason = (
                    f"ramp-up wait: {max(0.0, next_launch_time - now):.1f}s"
                )

            if block_reason is None:
                specification = pending.popleft()
                process, thread = _launch_worker(
                    specification,
                    python_executable=python_executable,
                    runner_script=runner_script,
                    project_root=project_root,
                    environment=environment,
                    creation_flags=creation_flags,
                    output_lock=output_lock,
                )
                active[process.pid] = (specification, process, thread)
                next_launch_time = time.monotonic() + config.ramp_up_seconds
                last_status = None
                if snapshot is None:
                    gpu_text = "telemetry=unavailable"
                else:
                    gpu_text = (
                        f"gpu={snapshot.utilization_percent:.0f}% "
                        f"free={snapshot.free_memory_mb:.0f}MiB "
                        f"temp={snapshot.temperature_c:.0f}C"
                    )
                print(
                    f"ELASTIC launch task={specification.index} "
                    f"label={specification.label} active={len(active)} "
                    f"pending={len(pending)} {gpu_text}",
                    flush=True,
                )
            elif block_reason != last_status or now - last_status_time >= 30.0:
                print(
                    f"ELASTIC waiting reason={block_reason} "
                    f"active={len(active)} pending={len(pending)}",
                    flush=True,
                )
                last_status = block_reason
                last_status_time = now

            time.sleep(config.poll_seconds)
    except KeyboardInterrupt:
        for _, process, _ in active.values():
            _terminate_process_tree(process)
        raise
    finally:
        for _, process, thread in active.values():
            process.wait()
            thread.join(timeout=5.0)

    print(
        f"ELASTIC phase={phase.protocol} complete success_tasks={succeeded} "
        f"failed_tasks={failed} retried_tasks={retried} skipped_tasks={skipped}",
        flush=True,
    )
    return 1 if failed else 0


def _run_phase(
    phase: ParallelPhase,
    *,
    python_executable: Path,
    runner_script: Path,
    project_root: Path,
    fail_fast: bool,
) -> int:
    print(
        f"PARALLEL phase={phase.protocol} workers={len(phase.workers)}",
        flush=True,
    )
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    processes = []
    output_threads = []
    output_lock = threading.Lock()
    creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    try:
        for specification in phase.workers:
            command = [
                str(python_executable),
                str(runner_script),
                "--parallel-worker",
                *specification.argv,
            ]
            print(
                f"PARALLEL worker={specification.index} label={specification.label}",
                flush=True,
            )
            process = subprocess.Popen(
                command,
                cwd=project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=environment,
                creationflags=creation_flags,
                start_new_session=os.name != "nt",
            )
            processes.append((specification, process))
            thread = threading.Thread(
                target=_stream_output,
                args=(process, f"[W{specification.index}]", output_lock),
                daemon=True,
            )
            thread.start()
            output_threads.append(thread)

        terminated = False
        while True:
            running = False
            for _, process in processes:
                return_code = process.poll()
                if return_code is None:
                    running = True
                elif return_code != 0 and fail_fast and not terminated:
                    terminated = True
                    for _, other in processes:
                        if other is not process:
                            _terminate_process_tree(other)
            if not running:
                break
            time.sleep(0.2)
    except KeyboardInterrupt:
        for _, process in processes:
            _terminate_process_tree(process)
        raise
    finally:
        for thread in output_threads:
            thread.join(timeout=5.0)

    return_codes = [process.wait() for _, process in processes]
    failures = sum(return_code != 0 for return_code in return_codes)
    print(
        f"PARALLEL phase={phase.protocol} complete "
        f"success_workers={len(return_codes) - failures} failed_workers={failures}",
        flush=True,
    )
    return 1 if failures else 0


def _summarize(selection: RunSelection) -> None:
    from cpdp.oversampling_validation.config import load_validation_config
    from cpdp.oversampling_validation.runner import OversamplingValidationRunner

    config = load_validation_config(selection.config_path)
    paths = OversamplingValidationRunner(config).summarize(selection.profile)
    for name, path in paths.items():
        print(f"{name}: {path}", flush=True)


def run_parallel_validation(
    argv: Sequence[str],
    *,
    workers: int,
    project_root: Path,
    python_executable: Path,
    runner_script: Path,
    elastic_gpu: Optional[ElasticGpuConfig] = None,
) -> int:
    selection = parse_run_selection(argv, project_root)
    if selection is None:
        raise ValueError("Parallel validation requires the run command")
    if selection.max_tasks is not None:
        raise ValueError("Parallel validation does not support --max-tasks")
    elastic_config = elastic_gpu or configured_elastic_gpu(workers)
    elastic_config.validate()
    phases = build_parallel_phases(
        argv,
        project_root,
        workers,
        atomic=elastic_config.enabled,
    )
    if not phases:
        raise ValueError("No validation tasks were selected")

    mode = "elastic" if elastic_config.enabled else "fixed"
    print(
        f"PARALLEL validation mode={mode} workers={workers} "
        f"profile={selection.profile} protocols={','.join(selection.protocols)}",
        flush=True,
    )
    for phase in phases:
        if elastic_config.enabled:
            return_code = _run_elastic_phase(
                phase,
                selection=selection,
                config=elastic_config,
                python_executable=python_executable,
                runner_script=runner_script,
                project_root=project_root,
            )
        else:
            return_code = _run_phase(
                phase,
                python_executable=python_executable,
                runner_script=runner_script,
                project_root=project_root,
                fail_fast=selection.fail_fast,
            )
        if return_code != 0:
            return return_code
    _summarize(selection)
    return 0
