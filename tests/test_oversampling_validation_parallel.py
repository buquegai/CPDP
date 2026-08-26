import json
import sys
from pathlib import Path

from cpdp.config import load_config
from oversampling_validation_parallel import (
    ElasticGpuConfig,
    GpuSnapshot,
    ParallelPhase,
    RunSelection,
    WorkerSpec,
    _is_transient_cuda_failure,
    _is_successful_result,
    _run_elastic_phase,
    build_parallel_phases,
    elastic_block_reason,
    parse_parallel_control,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _option_values(arguments, option):
    values = []
    for index, argument in enumerate(arguments[:-1]):
        if argument == option:
            values.append(arguments[index + 1])
    return values


def test_parallel_control_removes_private_arguments() -> None:
    control = parse_parallel_control(
        ("--parallel-worker", "run", "--workers", "3", "--profile", "full"),
        default_workers=4,
    )

    assert control.worker_process is True
    assert control.workers == 3
    assert control.argv == ("run", "--profile", "full")


def test_cpdp_parallel_shards_are_disjoint_and_complete() -> None:
    expected_seed_count = len(
        load_config(PROJECT_ROOT / "configs" / "experiment.yaml").seeds
    )
    phases = build_parallel_phases(
        (
            "run",
            "--profile",
            "full",
            "--protocol",
            "cpdp",
            "--variant",
            "smote",
            "--variant",
            "tabddpm",
            "--fail-fast",
        ),
        PROJECT_ROOT,
        workers=4,
    )

    assert len(phases) == 1
    assert len(phases[0].workers) == 4
    target_sets = [
        set(_option_values(worker.argv, "--target"))
        for worker in phases[0].workers
    ]
    assert sum(len(targets) for targets in target_sets) == 25
    assert len(set().union(*target_sets)) == 25
    assert all(
        left.isdisjoint(right)
        for index, left in enumerate(target_sets)
        for right in target_sets[index + 1 :]
    )
    assert all(
        len(_option_values(worker.argv, "--seed")) == expected_seed_count
        for worker in phases[0].workers
    )


def test_single_target_parallelizes_by_seed() -> None:
    expected_seed_count = len(
        load_config(PROJECT_ROOT / "configs" / "experiment.yaml").seeds
    )
    phases = build_parallel_phases(
        (
            "run",
            "--profile",
            "full",
            "--protocol",
            "cpdp",
            "--variant",
            "tabddpm",
            "--target",
            "NASA/CM1",
        ),
        PROJECT_ROOT,
        workers=4,
    )

    seed_sets = [
        set(_option_values(worker.argv, "--seed"))
        for worker in phases[0].workers
    ]
    assert len(seed_sets) == 4
    assert len(set().union(*seed_sets)) == expected_seed_count
    assert all(
        left.isdisjoint(right)
        for index, left in enumerate(seed_sets)
        for right in seed_sets[index + 1 :]
    )
    assert all(
        _option_values(worker.argv, "--target") == ["NASA/CM1"]
        for worker in phases[0].workers
    )


def test_cpdp_parallel_preserves_selected_seed_subset() -> None:
    selected_seeds = tuple(str(2026072200 + index) for index in range(1, 6))
    arguments = [
        "run",
        "--profile",
        "full",
        "--protocol",
        "cpdp",
        "--variant",
        "tabddpm",
    ]
    for seed in selected_seeds:
        arguments.extend(("--seed", seed))

    phases = build_parallel_phases(tuple(arguments), PROJECT_ROOT, workers=4)

    assert len(phases[0].workers) == 4
    assert all(
        tuple(_option_values(worker.argv, "--seed")) == selected_seeds
        for worker in phases[0].workers
    )


def test_source_cv_atomic_tasks_have_one_variant_seed_and_fold() -> None:
    selected_seeds = ("2026072201", "2026072202")
    phases = build_parallel_phases(
        (
            "run",
            "--profile",
            "full",
            "--protocol",
            "source_cv",
            "--variant",
            "tabddpm",
            "--family",
            "NASA",
            "--seed",
            selected_seeds[0],
            "--seed",
            selected_seeds[1],
        ),
        PROJECT_ROOT,
        workers=4,
        atomic=True,
    )

    assert len(phases) == 1
    assert len(phases[0].workers) == 6
    assert {
        tuple(_option_values(worker.argv, "--fold"))[0]
        for worker in phases[0].workers
    } == {"0", "1", "2"}
    assert all(
        len(_option_values(worker.argv, "--variant")) == 1
        and len(_option_values(worker.argv, "--family")) == 1
        and len(_option_values(worker.argv, "--seed")) == 1
        and len(_option_values(worker.argv, "--fold")) == 1
        and worker.result_path is not None
        for worker in phases[0].workers
    )


def test_cpdp_atomic_tasks_have_one_target_and_seed() -> None:
    phases = build_parallel_phases(
        (
            "run",
            "--profile",
            "full",
            "--protocol",
            "cpdp",
            "--variant",
            "tabddpm",
            "--target",
            "NASA/CM1",
            "--seed",
            "2026072201",
            "--seed",
            "2026072202",
        ),
        PROJECT_ROOT,
        workers=4,
        atomic=True,
    )

    assert len(phases[0].workers) == 2
    assert all(
        _option_values(worker.argv, "--variant") == ["tabddpm"]
        and _option_values(worker.argv, "--family") == ["NASA"]
        and _option_values(worker.argv, "--target") == ["NASA/CM1"]
        and len(_option_values(worker.argv, "--seed")) == 1
        and worker.result_path is not None
        for worker in phases[0].workers
    )


def _elastic_config() -> ElasticGpuConfig:
    return ElasticGpuConfig(
        enabled=True,
        max_workers=4,
        max_concurrent_tasks=2,
        launch_utilization_limit=75.0,
        minimum_free_memory_mb=2048.0,
        maximum_temperature_c=80.0,
        poll_seconds=2.0,
        ramp_up_seconds=8.0,
        pause_process_names=("game.exe",),
        transient_cuda_retries=3,
        retry_cooldown_seconds=30.0,
    )


def test_elastic_block_reason_accepts_available_gpu() -> None:
    snapshot = GpuSnapshot(42.0, 2048.0, 8192.0, 68.0)

    assert elastic_block_reason(snapshot, (), _elastic_config()) is None


def test_elastic_block_reason_prioritizes_pause_process() -> None:
    snapshot = GpuSnapshot(10.0, 1000.0, 8192.0, 55.0)

    reason = elastic_block_reason(snapshot, ("GAME.EXE",), _elastic_config())

    assert reason == "pause process active: game.exe"


def test_elastic_block_reason_checks_gpu_limits() -> None:
    config = _elastic_config()

    assert "temperature" in elastic_block_reason(
        GpuSnapshot(10.0, 1000.0, 8192.0, 80.0), (), config
    )
    assert "utilization" in elastic_block_reason(
        GpuSnapshot(75.0, 1000.0, 8192.0, 60.0), (), config
    )
    assert "free memory" in elastic_block_reason(
        GpuSnapshot(10.0, 7000.0, 8192.0, 60.0), (), config
    )


def test_successful_result_requires_matching_fingerprint(tmp_path) -> None:
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps({"status": "success", "fingerprint": "current"}),
        encoding="utf-8",
    )

    assert _is_successful_result(result_path, "current") is True
    assert _is_successful_result(result_path, "other") is False


def test_transient_cuda_failure_requires_known_driver_error(tmp_path) -> None:
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "status": "failure",
                "error_type": "RuntimeError",
                "error_message": "CUDA error: unspecified launch failure",
            }
        ),
        encoding="utf-8",
    )

    assert _is_transient_cuda_failure(result_path) is True

    result_path.write_text(
        json.dumps(
            {
                "status": "failure",
                "error_type": "RuntimeError",
                "error_message": (
                    "CUDA error: an illegal memory access was encountered"
                ),
            }
        ),
        encoding="utf-8",
    )
    assert _is_transient_cuda_failure(result_path) is True

    result_path.write_text(
        json.dumps(
            {
                "status": "failure",
                "error_type": "ValueError",
                "error_message": "invalid manifold configuration",
            }
        ),
        encoding="utf-8",
    )
    assert _is_transient_cuda_failure(result_path) is False


def test_elastic_phase_retries_transient_cuda_failure(
    monkeypatch, tmp_path: Path
) -> None:
    result_path = tmp_path / "result.json"
    attempt_path = tmp_path / "attempt.txt"
    runner_path = tmp_path / "fake_runner.py"
    runner_path.write_text(
        "\n".join(
            (
                "import json",
                "from pathlib import Path",
                f"result_path = Path({str(result_path)!r})",
                f"attempt_path = Path({str(attempt_path)!r})",
                "attempt = int(attempt_path.read_text() or '0') "
                "+ 1 if attempt_path.exists() else 1",
                "attempt_path.write_text(str(attempt), encoding='utf-8')",
                "if attempt == 1:",
                "    result_path.write_text(json.dumps({",
                "        'status': 'failure',",
                "        'fingerprint': 'fingerprint',",
                "        'error_type': 'RuntimeError',",
                "        'error_message': 'CUDA error: unspecified launch failure',",
                "    }), encoding='utf-8')",
                "    raise SystemExit(1)",
                "result_path.write_text(json.dumps({",
                "    'status': 'success',",
                "    'fingerprint': 'fingerprint',",
                "}), encoding='utf-8')",
            )
        ),
        encoding="utf-8",
    )
    selection = RunSelection(
        config_path=tmp_path / "config.yaml",
        profile="search",
        protocols=("source_cv",),
        variants=("tabddpm",),
        families=("AEEEM",),
        targets=(),
        seeds=(1,),
        folds=(0,),
        max_tasks=None,
        fail_fast=True,
        force=False,
        fingerprint="fingerprint",
        result_root=tmp_path,
    )
    phase = ParallelPhase(
        "source_cv",
        (WorkerSpec(1, "transient-cuda", (), result_path),),
    )
    config = ElasticGpuConfig(
        enabled=True,
        max_workers=2,
        max_concurrent_tasks=2,
        launch_utilization_limit=75.0,
        minimum_free_memory_mb=2048.0,
        maximum_temperature_c=80.0,
        poll_seconds=0.01,
        ramp_up_seconds=0.0,
        pause_process_names=(),
        transient_cuda_retries=1,
        retry_cooldown_seconds=0.0,
    )
    monkeypatch.setattr(
        "oversampling_validation_parallel.query_nvidia_gpu",
        lambda: GpuSnapshot(0.0, 1000.0, 8192.0, 50.0),
    )

    return_code = _run_elastic_phase(
        phase,
        selection=selection,
        config=config,
        python_executable=Path(sys.executable),
        runner_script=runner_path,
        project_root=tmp_path,
    )

    assert return_code == 0
    assert attempt_path.read_text(encoding="utf-8") == "2"
    assert _is_successful_result(result_path, "fingerprint") is True
