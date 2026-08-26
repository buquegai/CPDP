from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml

from cpdp.oversampling.types import AugmentationResult, freeze_array
from cpdp.types import LabeledProject, ProjectFeatures, UnlabeledProject


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


def mapping_signature(values: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _jsonable(values), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_data_signature(sources: Sequence[LabeledProject]) -> str:
    digest = hashlib.sha256()
    for project in sorted(sources, key=lambda item: item.key):
        digest.update(project.key.encode("utf-8"))
        digest.update("\0".join(project.features.feature_names).encode("utf-8"))
        for array in (
            project.features.values,
            project.labels,
            project.features.effort,
            project.features.row_ids,
        ):
            contiguous = np.ascontiguousarray(array)
            digest.update(str(contiguous.dtype).encode("ascii"))
            digest.update(str(contiguous.shape).encode("ascii"))
            digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(_jsonable(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(temporary, path)


def save_augmentation_cache(
    cache_path: Path,
    augmentation: AugmentationResult,
    *,
    target_key: str,
    oversampling_seed: int,
    oversampling_config: Mapping[str, Any],
    sources: Sequence[LabeledProject],
) -> Mapping[str, Any]:
    cache_path = cache_path.resolve()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    arrays: dict[str, np.ndarray] = {}
    projects = []
    for index, project in enumerate(augmentation.projects):
        prefix = f"project_{index}"
        arrays[f"{prefix}_values"] = project.features.values
        arrays[f"{prefix}_labels"] = project.labels
        arrays[f"{prefix}_effort"] = project.features.effort
        arrays[f"{prefix}_row_ids"] = project.features.row_ids
        projects.append(
            {
                "prefix": prefix,
                "family": project.features.family,
                "name": project.features.name,
                "feature_names": list(project.features.feature_names),
                "effort_name": project.features.effort_name,
                "cleaning": dict(project.features.cleaning),
            }
        )
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, cache_path)

    manifest = {
        "schema_version": 1,
        "target": target_key,
        "oversampling_seed": int(oversampling_seed),
        "oversampling_config_signature": mapping_signature(oversampling_config),
        "source_data_signature": source_data_signature(sources),
        "source_projects": [project.key for project in sources],
        "projects": projects,
        "augmentation_metadata": augmentation.metadata,
        "archive": cache_path.name,
    }
    _atomic_json(cache_path.with_suffix(".json"), manifest)
    return manifest


def load_augmentation_cache(
    cache_path: Path,
    *,
    target_key: str,
    oversampling_seed: int,
    oversampling_config: Mapping[str, Any],
    sources: Sequence[LabeledProject],
) -> AugmentationResult:
    cache_path = cache_path.resolve()
    manifest_path = cache_path.with_suffix(".json")
    if not cache_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"IPGDO augmentation cache is incomplete: {cache_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    expected = {
        "target": target_key,
        "oversampling_seed": int(oversampling_seed),
        "oversampling_config_signature": mapping_signature(oversampling_config),
        "source_data_signature": source_data_signature(sources),
        "source_projects": [project.key for project in sources],
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"IPGDO cache validation failed for {target_key}: {key} changed"
            )

    projects = []
    with np.load(cache_path, allow_pickle=False) as archive:
        for item in manifest["projects"]:
            prefix = item["prefix"]
            features = ProjectFeatures(
                family=str(item["family"]),
                name=str(item["name"]),
                values=freeze_array(archive[f"{prefix}_values"], np.float64),
                feature_names=tuple(str(name) for name in item["feature_names"]),
                effort=freeze_array(archive[f"{prefix}_effort"], np.float64),
                effort_name=str(item["effort_name"]),
                row_ids=freeze_array(archive[f"{prefix}_row_ids"], np.int64),
                cleaning=dict(item.get("cleaning", {})),
            )
            projects.append(
                LabeledProject(
                    features=features,
                    labels=freeze_array(archive[f"{prefix}_labels"], np.int64),
                )
            )
    metadata = dict(manifest.get("augmentation_metadata", {}))
    metadata["cache"] = {
        "reused": True,
        "path": str(cache_path),
        "target": target_key,
        "oversampling_seed": int(oversampling_seed),
    }
    return AugmentationResult(projects=tuple(projects), metadata=metadata)


def _resolve_path(value: str, base: Path) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(value)))
    return path if path.is_absolute() else (base / path).resolve()


def load_selected_augmentation(
    project_root: Path,
    target: UnlabeledProject,
    sources: Sequence[LabeledProject],
    oversampling_config: Mapping[str, Any],
    reuse_config: Mapping[str, Any],
) -> Tuple[Optional[AugmentationResult], Optional[int]]:
    if not bool(reuse_config.get("enabled", False)):
        return None, None
    seed_map_path = _resolve_path(
        str(reuse_config.get("seed_map", "outputs/ipgdo_seed_search/best_oversampling_seeds.yaml")),
        project_root,
    )
    if not seed_map_path.is_file():
        raise FileNotFoundError(f"IPGDO best-seed map not found: {seed_map_path}")
    with seed_map_path.open("r", encoding="utf-8") as handle:
        seed_map = yaml.safe_load(handle)
    entries = seed_map.get("targets", {}) if isinstance(seed_map, Mapping) else {}
    entry = entries.get(target.key)
    if not isinstance(entry, Mapping):
        raise KeyError(f"No selected IPGDO seed for target {target.key}")
    selected_seed = int(entry["best_seed"])
    cache_path = _resolve_path(str(entry["cache_file"]), seed_map_path.parent)
    augmentation = load_augmentation_cache(
        cache_path,
        target_key=target.key,
        oversampling_seed=selected_seed,
        oversampling_config=oversampling_config,
        sources=sources,
    )
    return augmentation, selected_seed
