import json
import os
from pathlib import Path

import pandas as pd

from parameter_search_runtime_patch import (
    _replace_with_retry,
    _resume_source_signature,
    install_parameter_search_runtime_patch,
)
from cpdp.oversampling_validation.store import ValidationStore


def test_resume_signature_is_recovered_for_any_interrupted_search(tmp_path: Path) -> None:
    state_path = tmp_path / "search_state.json"
    state_path.write_text(
        json.dumps(
            {
                "status": "failed",
                "last_error": "RuntimeError: CUDA launch failure",
                "trials": {
                    "trial": {
                        "config_path": str(
                            tmp_path / "trial__code_52c3306fb6c8.yaml"
                        )
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    assert _resume_source_signature(state_path) == "52c3306fb6c8"

    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["status"] = "interrupted"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    assert _resume_source_signature(state_path) == "52c3306fb6c8"

    state["status"] = "completed"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    assert _resume_source_signature(state_path) is None


def test_runtime_patch_reuses_complete_summary_without_rewriting(tmp_path: Path) -> None:
    state_path = tmp_path / "missing_state.json"
    assert (
        install_parameter_search_runtime_patch(tmp_path, state_path=state_path) is None
    )
    store = ValidationStore(tmp_path / "results", "fingerprint", "search")
    run_path = store.root / "runs" / "result.json"
    run_path.parent.mkdir(parents=True)
    run_path.write_text(json.dumps({"status": "success"}), encoding="utf-8")

    summary_dir = store.root / "summary"
    summary_dir.mkdir(parents=True)
    csv_names = (
        "utility_runs.csv",
        "quality_runs.csv",
        "project_utility_runs.csv",
        "utility_summary.csv",
    )
    for name in csv_names:
        pd.DataFrame({"value": [name]}).to_csv(summary_dir / name, index=False)
    manifest_path = summary_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "fingerprint": "fingerprint",
                "profile": "search",
                "successful_runs": 1,
            }
        ),
        encoding="utf-8",
    )
    before = {
        path.name: path.read_bytes() for path in summary_dir.iterdir() if path.is_file()
    }

    paths = store.summarize()

    assert paths["manifest"] == manifest_path
    after = {
        path.name: path.read_bytes() for path in summary_dir.iterdir() if path.is_file()
    }
    assert after == before


def test_replace_retries_transient_permission_error(monkeypatch, tmp_path: Path) -> None:
    temporary = tmp_path / "source.tmp"
    destination = tmp_path / "destination.csv"
    temporary.write_text("new", encoding="utf-8")
    destination.write_text("old", encoding="utf-8")
    attempts = 0
    original_replace = os.replace

    def flaky_replace(source: Path, target: Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError("temporarily locked")
        original_replace(source, target)

    monkeypatch.setattr("parameter_search_runtime_patch.os.replace", flaky_replace)
    monkeypatch.setattr("parameter_search_runtime_patch.time.sleep", lambda _: None)

    _replace_with_retry(temporary, destination)

    assert attempts == 3
    assert destination.read_text(encoding="utf-8") == "new"
