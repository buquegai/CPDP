from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from cpdp.oversampling_validation.config import OversamplingValidationConfig
from cpdp.oversampling_validation.strategies import effective_strategy_config


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
    return value


def _update_file(digest: Any, path: Path, relative_to: Path) -> None:
    digest.update(str(path.relative_to(relative_to)).replace("\\", "/").encode("utf-8"))
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)


def experiment_fingerprint(
    config: OversamplingValidationConfig,
    profile_name: str,
    base_oversampling_config: Mapping[str, Any],
) -> str:
    profile = config.profile(profile_name)
    payload = config.effective_mapping(profile_name)
    payload["effective_oversampling"] = {
        name: effective_strategy_config(
            config.variant(name).kind,
            base_oversampling_config,
            config.variant(name).overrides,
        )
        for name in profile.variants
    }
    digest = hashlib.sha256(
        json.dumps(
            _jsonable(payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    project_root = config.source_path.parent.parent
    source_roots = (
        project_root / "src" / "cpdp" / "oversampling",
        project_root / "src" / "cpdp" / "oversampling_validation",
        project_root / "src" / "cpdp" / "predictors",
    )
    for source_root in source_roots:
        for path in sorted(source_root.rglob("*.py")):
            _update_file(digest, path, project_root)
    for family in config.experiment.data.families:
        for project in family.projects:
            path = (
                config.experiment.data.root
                / family.name
                / f"{project}{config.experiment.data.file_suffix}"
            )
            if not path.is_file():
                raise FileNotFoundError(f"Dataset file not found while fingerprinting: {path}")
            _update_file(digest, path, config.experiment.data.root)
    return digest.hexdigest()[:20]
