from __future__ import annotations

import json
import os
import re
import time
import uuid
from functools import wraps
from pathlib import Path
from typing import Any, Mapping, Optional


_SOURCE_SIGNATURE_PATTERN = re.compile(r"__code_([0-9a-f]{12})\.yaml$")
_REPLACE_DELAYS = (0.0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2)


def _summary_paths(store: Any) -> dict[str, Path]:
    summary_dir = store.root / "summary"
    return {
        "utility_runs": summary_dir / "utility_runs.csv",
        "quality_runs": summary_dir / "quality_runs.csv",
        "project_utility_runs": summary_dir / "project_utility_runs.csv",
        "utility_summary": summary_dir / "utility_summary.csv",
        "manifest": summary_dir / "manifest.json",
    }


def _successful_record_count(store: Any) -> int:
    return sum(1 for _ in store._records())


def _summary_is_current(store: Any, paths: Mapping[str, Path]) -> bool:
    if not all(path.is_file() for path in paths.values()):
        return False
    try:
        with paths["manifest"].open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        return (
            manifest.get("fingerprint") == store.fingerprint
            and manifest.get("profile") == store.profile
            and int(manifest.get("successful_runs", -1))
            == _successful_record_count(store)
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")


def _replace_with_retry(temporary: Path, destination: Path) -> None:
    for attempt, delay in enumerate(_REPLACE_DELAYS):
        if delay:
            time.sleep(delay)
        try:
            os.replace(temporary, destination)
            return
        except PermissionError:
            if attempt == len(_REPLACE_DELAYS) - 1:
                raise


def _discard_temporary(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    from cpdp.oversampling_validation.store import _jsonable

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                _jsonable(payload),
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        _replace_with_retry(temporary, path)
    finally:
        _discard_temporary(temporary)


def _atomic_csv(path: Path, frame: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8-sig")
        _replace_with_retry(temporary, path)
    finally:
        _discard_temporary(temporary)


def _resume_source_signature(state_path: Path) -> Optional[str]:
    if not state_path.is_file():
        return None
    try:
        with state_path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if state.get("status") not in {"failed", "running", "interrupted"}:
        return None
    signatures = set()
    for trial in state.get("trials", {}).values():
        match = _SOURCE_SIGNATURE_PATTERN.search(str(trial.get("config_path", "")))
        if match:
            signatures.add(match.group(1))
    return signatures.pop() if len(signatures) == 1 else None


def _install_store_patch() -> None:
    from cpdp.oversampling_validation.store import ValidationStore

    if getattr(ValidationStore, "_runtime_file_lock_patch", False):
        return
    original_summarize = ValidationStore.summarize

    @wraps(original_summarize)
    def summarize(store: Any) -> Mapping[str, Path]:
        paths = _summary_paths(store)
        if _summary_is_current(store, paths):
            return paths
        return original_summarize(store)

    ValidationStore._atomic_json = staticmethod(_atomic_json)
    ValidationStore._atomic_csv = staticmethod(_atomic_csv)
    ValidationStore.summarize = summarize
    ValidationStore._runtime_file_lock_patch = True


def _install_source_signature_override(
    project_root: Path, resume_signature: Optional[str]
) -> None:
    if resume_signature is None:
        return
    from cpdp.oversampling_validation import parameter_search

    if hasattr(parameter_search, "_runtime_original_source_signature"):
        return
    original = parameter_search._source_signature
    expected_root = project_root.resolve()

    @wraps(original)
    def source_signature(candidate_root: Path) -> str:
        if Path(candidate_root).resolve() == expected_root:
            return resume_signature
        return original(candidate_root)

    parameter_search._runtime_original_source_signature = original
    parameter_search._source_signature = source_signature


def install_parameter_search_runtime_patch(
    project_root: Path, *, state_path: Path
) -> Optional[str]:
    resume_signature = _resume_source_signature(state_path)
    _install_store_patch()
    _install_source_signature_override(project_root, resume_signature)
    return resume_signature


__all__ = ["install_parameter_search_runtime_patch"]
