from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import yaml

from cpdp.config import ExperimentConfig, load_config


def _expand_path(value: str, base: Path) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(value))
    path = Path(expanded)
    return path if path.is_absolute() else (base / path).resolve()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


@dataclass(frozen=True)
class VariantConfig:
    name: str
    kind: str
    overrides: Mapping[str, Any]


@dataclass(frozen=True)
class ClassifierConfig:
    kind: str = "logistic_regression"
    c: float = 1.0
    max_iter: int = 2000
    class_weight: Optional[str] = None
    method: Optional[str] = None
    predictor: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.kind == "logistic_regression":
            if self.c <= 0:
                raise ValueError("classifier.c must be positive")
            if self.max_iter < 1:
                raise ValueError("classifier.max_iter must be positive")
            if self.class_weight not in {None, "balanced"}:
                raise ValueError("classifier.class_weight must be null or 'balanced'")
            return
        if self.kind == "formal_kan_jepa_tabr":
            if self.method != "ipgdo_kan_jepa":
                raise ValueError(
                    "formal_kan_jepa_tabr requires classifier.method='ipgdo_kan_jepa'"
                )
            if not self.predictor:
                raise ValueError("formal predictor configuration cannot be empty")
            return
        raise ValueError(
            "classifier.kind must be 'logistic_regression' or "
            "'formal_kan_jepa_tabr'"
        )


@dataclass(frozen=True)
class QualityConfig:
    max_samples: int = 1000
    knn_neighbors: int = 5
    near_duplicate_quantile: float = 0.05
    manifold_lower_quantile: float = 0.05
    manifold_upper_quantile: float = 0.95
    coverage_quantile: float = 0.95
    exact_tolerance: float = 1e-10

    def validate(self) -> None:
        if self.max_samples < 2:
            raise ValueError("quality.max_samples must be at least 2")
        if self.knn_neighbors < 1:
            raise ValueError("quality.knn_neighbors must be positive")
        for name in (
            "near_duplicate_quantile",
            "manifold_lower_quantile",
            "manifold_upper_quantile",
            "coverage_quantile",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"quality.{name} must be in [0, 1]")
        if self.manifold_lower_quantile >= self.manifold_upper_quantile:
            raise ValueError("quality manifold quantiles must be increasing")
        if self.exact_tolerance < 0:
            raise ValueError("quality.exact_tolerance cannot be negative")


@dataclass(frozen=True)
class SourceCVConfig:
    enabled: bool = True
    folds: int = 3
    validation_fraction: float = 0.2

    def validate(self) -> None:
        if self.folds < 2:
            raise ValueError("source_cv.folds must be at least 2")
        if not 0.0 < self.validation_fraction < 0.5:
            raise ValueError("source_cv.validation_fraction must be in (0, 0.5)")


@dataclass(frozen=True)
class CPDPConfig:
    enabled: bool = True
    validation_fraction: float = 0.2

    def validate(self) -> None:
        if not 0.0 < self.validation_fraction < 0.5:
            raise ValueError("cpdp.validation_fraction must be in (0, 0.5)")


@dataclass(frozen=True)
class ValidationProfile:
    name: str
    oversampling_profile: str
    seeds: Tuple[int, ...]
    variants: Tuple[str, ...]
    source_cv: SourceCVConfig
    cpdp: CPDPConfig
    classifier: ClassifierConfig
    quality: QualityConfig

    def validate(self) -> None:
        if not self.seeds:
            raise ValueError(f"profiles.{self.name}.seeds cannot be empty")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError(f"profiles.{self.name}.seeds must be unique")
        if not self.variants:
            raise ValueError(f"profiles.{self.name}.variants cannot be empty")
        self.source_cv.validate()
        self.cpdp.validate()
        self.classifier.validate()
        self.quality.validate()


@dataclass(frozen=True)
class OversamplingValidationConfig:
    source_path: Path
    experiment: ExperimentConfig
    output_dir: Path
    oversampling_method: str
    variants: Mapping[str, VariantConfig]
    profiles: Mapping[str, ValidationProfile]

    def profile(self, name: str) -> ValidationProfile:
        try:
            return self.profiles[name]
        except KeyError as exc:
            raise KeyError(f"Unknown validation profile: {name}") from exc

    def variant(self, name: str) -> VariantConfig:
        try:
            return self.variants[name]
        except KeyError as exc:
            raise KeyError(f"Unknown oversampling validation variant: {name}") from exc

    def effective_mapping(self, profile_name: str) -> Dict[str, Any]:
        profile = self.profile(profile_name)
        return {
            "source_path": str(self.source_path),
            "experiment_config": str(self.experiment.source_path),
            "data": {
                "root": str(self.experiment.data.root),
                "file_suffix": self.experiment.data.file_suffix,
                "label_column": self.experiment.data.label_column,
                "cleaning": self.experiment.data.cleaning,
                "families": [
                    {
                        "name": family.name,
                        "projects": list(family.projects),
                        "effort_column": family.effort_column,
                    }
                    for family in self.experiment.data.families
                ],
            },
            "output_dir": str(self.output_dir),
            "oversampling_method": self.oversampling_method,
            "profile": asdict(profile),
            "variants": {
                name: asdict(self.variant(name)) for name in profile.variants
            },
        }


def _build_profile(
    name: str,
    values: Mapping[str, Any],
    experiment: ExperimentConfig,
) -> ValidationProfile:
    oversampling_profile = str(values.get("oversampling_profile", name))
    source_cv = SourceCVConfig(**_mapping(values.get("source_cv"), "source_cv"))
    cpdp = CPDPConfig(**_mapping(values.get("cpdp"), "cpdp"))
    classifier_values = dict(_mapping(values.get("classifier"), "classifier"))
    classifier_kind = str(classifier_values.get("kind", "logistic_regression"))
    if classifier_kind == "formal_kan_jepa_tabr":
        unknown = sorted(set(classifier_values) - {"kind", "method"})
        if unknown:
            raise ValueError(
                "Unsupported formal classifier option(s): " + ", ".join(unknown)
            )
        method = str(classifier_values.get("method", "ipgdo_kan_jepa"))
        method_params = experiment.method_params(oversampling_profile, method)
        predictor = _mapping(
            method_params.get("predictor"),
            f"{oversampling_profile}/{method}.predictor",
        )
        classifier = ClassifierConfig(
            kind=classifier_kind,
            method=method,
            predictor=deepcopy(dict(predictor)),
        )
    else:
        classifier = ClassifierConfig(**classifier_values)
    quality = QualityConfig(**_mapping(values.get("quality"), "quality"))
    configured_seeds = values.get("seeds", ())
    seeds = (
        experiment.seeds
        if isinstance(configured_seeds, str)
        and configured_seeds.lower() == "experiment"
        else tuple(int(seed) for seed in configured_seeds)
    )
    profile = ValidationProfile(
        name=name,
        oversampling_profile=oversampling_profile,
        seeds=seeds,
        variants=tuple(str(variant) for variant in values.get("variants", ())),
        source_cv=source_cv,
        cpdp=cpdp,
        classifier=classifier,
        quality=quality,
    )
    profile.validate()
    return profile


def load_validation_config(path: Path) -> OversamplingValidationConfig:
    source_path = path.resolve()
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if raw.get("schema_version") != 1:
        raise ValueError("Unsupported oversampling validation schema_version")

    experiment_path = _expand_path(
        str(raw.get("experiment_config", "experiment.yaml")), source_path.parent
    )
    experiment = load_config(experiment_path)
    variants = {
        str(name): VariantConfig(
            name=str(name),
            kind=str(values.get("kind", name)),
            overrides=dict(_mapping(values.get("overrides"), f"variants.{name}.overrides")),
        )
        for name, values in _mapping(raw.get("variants"), "variants").items()
    }
    profiles = {
        str(name): _build_profile(
            str(name),
            _mapping(values, f"profiles.{name}"),
            experiment,
        )
        for name, values in _mapping(raw.get("profiles"), "profiles").items()
    }
    if not variants:
        raise ValueError("At least one validation variant must be configured")
    if not profiles:
        raise ValueError("At least one validation profile must be configured")
    for profile in profiles.values():
        unknown = sorted(set(profile.variants) - set(variants))
        if unknown:
            raise ValueError(
                f"Profile {profile.name} references unknown variants: {', '.join(unknown)}"
            )
    project_root = source_path.parent.parent
    return OversamplingValidationConfig(
        source_path=source_path,
        experiment=experiment,
        output_dir=_expand_path(
            str(raw.get("output_dir", "outputs/oversampling_validation")),
            project_root,
        ),
        oversampling_method=str(
            raw.get("oversampling_method", "ipgdo_kan_jepa")
        ),
        variants=variants,
        profiles=profiles,
    )
