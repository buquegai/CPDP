from pathlib import Path

import pytest

from cpdp.launcher import (
    ALL_SEEDS,
    LaunchOptions,
    available_seed_choices,
    build_run_command,
    default_seed_choice,
    resolve_seed_choice,
    seed_count_choice,
)
from cpdp.methods.registry import METHOD_NAMES


def test_launcher_builds_all_method_batch_command() -> None:
    command = build_run_command(
        "python.exe",
        Path("D:/CPDP"),
        LaunchOptions(
            methods=METHOD_NAMES,
            profile="full",
            target=None,
            seeds=None,
            max_runs=10,
            workers=3,
            force=False,
        ),
    )

    assert command[3:7] == ["--method", "all", "--profile", "full"]
    assert command[-4:] == ["--max-runs", "10", "--workers", "3"]


def test_launcher_builds_single_target_seed_command() -> None:
    command = build_run_command(
        "python.exe",
        Path("D:/CPDP"),
        LaunchOptions(
            methods=("master",),
            profile="smoke",
            target="ReLink/Apache",
            seeds=(2026072201,),
            max_runs=None,
            workers=1,
            force=True,
        ),
    )

    assert ["--method", "master"] == command[3:5]
    assert ["--target", "ReLink/Apache"] == command[7:9]
    assert ["--seed", "2026072201"] == command[9:11]
    assert command[-3:] == ["--workers", "1", "--force"]


def test_launcher_rejects_invalid_parallel_worker_count() -> None:
    with pytest.raises(ValueError, match="Parallel workers"):
        build_run_command(
            "python.exe",
            Path("D:/CPDP"),
            LaunchOptions(
                methods=("master",),
                profile="smoke",
                target="ReLink/Apache",
                seeds=(2026072201,),
                max_runs=1,
                workers=0,
                force=False,
            ),
        )


def test_launcher_uses_all_seeds_for_full_profile() -> None:
    seeds = (2026072201, 2026072202, 2026072203)

    assert default_seed_choice("smoke", seeds) == "2026072201"
    assert default_seed_choice("full", seeds) == ALL_SEEDS


def test_launcher_exposes_and_resolves_seed_count_choices() -> None:
    seeds = tuple(range(2026072201, 2026072231))

    choices = available_seed_choices(seeds)

    assert choices[:4] == (
        ALL_SEEDS,
        seed_count_choice(5),
        seed_count_choice(10),
        seed_count_choice(20),
    )
    assert resolve_seed_choice(seed_count_choice(5), seeds) == seeds[:5]
    assert resolve_seed_choice(seed_count_choice(10), seeds) == seeds[:10]
    assert resolve_seed_choice(str(seeds[7]), seeds) == (seeds[7],)
    assert resolve_seed_choice(ALL_SEEDS, seeds) is None


def test_launcher_builds_repeated_seed_arguments() -> None:
    seeds = (2026072201, 2026072202, 2026072203)
    command = build_run_command(
        "python.exe",
        Path("D:/CPDP"),
        LaunchOptions(
            methods=("master",),
            profile="full",
            target=None,
            seeds=seeds,
            max_runs=None,
            workers=1,
            force=False,
        ),
    )

    assert command.count("--seed") == len(seeds)
    for seed in seeds:
        index = command.index(str(seed))
        assert command[index - 1] == "--seed"
