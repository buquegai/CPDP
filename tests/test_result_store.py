import json

import pandas as pd
import pytest

from cpdp.metrics import METRIC_NAMES
from cpdp.result_store import ResultStore


def test_summary_uses_run_means_then_project_macro_means(tmp_path) -> None:
    store = ResultStore(tmp_path, save_predictions=False)
    for target, seed, auc in (
        ("A", 1, 0.6),
        ("A", 2, 0.8),
        ("B", 1, 0.4),
        ("B", 2, 0.6),
    ):
        payload = {
            "status": "success",
            "profile": "full",
            "method": "demo",
            "family": "family",
            "target": target,
            "seed": seed,
            "threshold": 0.5,
            "duration_seconds": 1.0,
            "metrics": {metric: auc for metric in METRIC_NAMES},
        }
        path = store.run_path("full", "demo", "family", target, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    paths = store.summarize(expected_repetitions=2)
    targets = pd.read_csv(paths["targets"])
    overall = pd.read_csv(paths["overall"])

    target_a = targets.loc[targets["target"] == "A"].iloc[0]
    assert target_a["auc_mean"] == pytest.approx(0.7)
    assert target_a["auc_count"] == 2
    assert bool(target_a["complete"])
    assert overall.iloc[0]["auc_mean"] == pytest.approx(0.6)
    assert overall.iloc[0]["target_count"] == 2
    assert (
        tmp_path
        / "summary"
        / "by_method"
        / "full"
        / "demo"
        / "demo_target_means.csv"
    ).is_file()
    assert paths["history_snapshot"].joinpath("target_means.csv").is_file()


def test_smoke_and_full_results_never_collide(tmp_path) -> None:
    store = ResultStore(tmp_path, save_predictions=False)

    assert store.run_path("smoke", "demo", "family", "A", 1) != store.run_path(
        "full", "demo", "family", "A", 1
    )
    assert not store.is_complete("full", "demo", "family", "A", 1)


def test_summary_excludes_seeds_outside_current_experiment(tmp_path) -> None:
    store = ResultStore(tmp_path, save_predictions=False)
    for seed, auc in ((1, 0.6), (2, 0.8), (3, 1.0)):
        payload = {
            "status": "success",
            "profile": "full",
            "method": "demo",
            "family": "family",
            "target": "A",
            "seed": seed,
            "threshold": 0.5,
            "duration_seconds": 1.0,
            "metrics": {metric: auc for metric in METRIC_NAMES},
        }
        path = store.run_path("full", "demo", "family", "A", seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    paths = store.summarize(expected_repetitions=2, expected_seeds=(1, 2))
    runs = pd.read_csv(paths["runs"])
    targets = pd.read_csv(paths["targets"])

    assert set(runs["seed"]) == {1, 2}
    assert targets.iloc[0]["auc_mean"] == pytest.approx(0.7)
    assert targets.iloc[0]["auc_count"] == 2
    assert bool(targets.iloc[0]["complete"])


def test_result_signature_invalidates_an_old_success(tmp_path) -> None:
    store = ResultStore(tmp_path, save_predictions=False)
    path = store.run_path("full", "fedcpdp", "family", "A", 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "success",
                "method_params": {"result_signature": "old"},
            }
        ),
        encoding="utf-8",
    )

    assert store.is_complete("full", "fedcpdp", "family", "A", 1)
    assert not store.is_complete(
        "full",
        "fedcpdp",
        "family",
        "A",
        1,
        result_signature="new",
    )
    assert store.is_complete(
        "full",
        "fedcpdp",
        "family",
        "A",
        1,
        result_signature="old",
    )


def test_each_summary_call_keeps_a_distinct_history_snapshot(tmp_path) -> None:
    store = ResultStore(tmp_path, save_predictions=False)
    payload = {
        "status": "success",
        "profile": "smoke",
        "method": "demo",
        "family": "family",
        "target": "A",
        "seed": 1,
        "threshold": 0.5,
        "duration_seconds": 1.0,
        "metrics": {metric: 0.5 for metric in METRIC_NAMES},
    }
    path = store.run_path("smoke", "demo", "family", "A", 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")

    first = store.summarize()["history_snapshot"]
    second = store.summarize()["history_snapshot"]

    assert first != second
    assert first.joinpath("run_results.csv").is_file()
    assert second.joinpath("run_results.csv").is_file()
