from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

import yaml


LEGACY_METHOD_ALIASES = {
    "apc_pc_tabddpm_kan_jepa": "ipgdo_kan_jepa",
    "apc_pc_tabddpm_transformer_jepa": "ipgdo_transformer_jepa",
}


@dataclass(frozen=True)
class FamilyConfig:
    name: str
    projects: Tuple[str, ...]
    effort_column: str


@dataclass(frozen=True)
class DataConfig:
    root: Path
    file_suffix: str
    label_column: str
    cleaning: str
    families: Tuple[FamilyConfig, ...]


@dataclass(frozen=True)
class ExperimentConfig:
    source_path: Path
    data: DataConfig
    protocol: str
    target_labels: str
    repetitions: int
    seeds: Tuple[int, ...]
    metrics: Tuple[str, ...]
    output_dir: Path
    save_predictions: bool
    profiles: Mapping[str, Mapping[str, Mapping[str, Any]]]

    def family(self, name: str) -> FamilyConfig:
        for family in self.data.families:
            if family.name.lower() == name.lower():
                return family
        raise KeyError(f"Unknown dataset family: {name}")

    def method_params(self, profile: str, method: str) -> Dict[str, Any]:
        configured_method = LEGACY_METHOD_ALIASES.get(method, method)
        try:
            values = self.profiles[profile][configured_method]
        except KeyError as exc:
            raise KeyError(f"No profile configuration for {profile}/{method}") from exc
        return dict(values)

    def targets(self) -> Iterable[Tuple[str, str]]:
        for family in self.data.families:
            for project in family.projects:
                yield family.name, project


def _expand_path(value: str, base: Path) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(value))
    path = Path(expanded)
    return path if path.is_absolute() else (base / path).resolve()


def load_config(path: Path) -> ExperimentConfig:
    source_path = path.resolve()
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if raw.get("schema_version") != 1:
        raise ValueError("Unsupported configuration schema_version")

    data_raw = raw["data"]
    families = tuple(
        FamilyConfig(
            name=name,
            projects=tuple(str(project) for project in values["projects"]),
            effort_column=str(values["effort_column"]),
        )
        for name, values in data_raw["families"].items()
    )
    data = DataConfig(
        root=_expand_path(str(data_raw["root"]), source_path.parent),
        file_suffix=str(data_raw["file_suffix"]),
        label_column=str(data_raw["label_column"]),
        cleaning=str(data_raw.get("cleaning", "none")),
        families=families,
    )

    experiment = raw["experiment"]
    repetitions = int(experiment["repetitions"])
    seeds = tuple(int(seed) for seed in experiment["seeds"])
    if len(seeds) != repetitions:
        raise ValueError(
            f"Expected {repetitions} seeds, but configuration contains {len(seeds)}"
        )
    if len(set(seeds)) != len(seeds):
        raise ValueError("Experiment seeds must be unique")

    project_root = source_path.parent.parent
    return ExperimentConfig(
        source_path=source_path,
        data=data,
        protocol=str(experiment["protocol"]),
        target_labels=str(experiment["target_labels"]),
        repetitions=repetitions,
        seeds=seeds,
        metrics=tuple(str(metric) for metric in experiment["metrics"]),
        output_dir=_expand_path(str(experiment["output_dir"]), project_root),
        save_predictions=bool(experiment.get("save_predictions", True)),
        profiles=raw["profiles"],
    )
