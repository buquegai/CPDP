from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from cpdp.metrics import METRIC_NAMES, evaluate_predictions
from cpdp.oversampling.types import freeze_array
from cpdp.oversampling_validation.classifier import (
    FittedValidationClassifier,
    fit_validation_classifier,
    predict_validation_target,
)
from cpdp.oversampling_validation.config import ValidationProfile
from cpdp.oversampling_validation.quality import evaluate_synthetic_quality
from cpdp.oversampling_validation.strategies import OversamplingStrategy
from cpdp.types import LabeledProject, ProjectFeatures


@dataclass(frozen=True)
class ProtocolResult:
    utility: Mapping[str, Any]
    quality: Tuple[Mapping[str, Any], ...]


def _derived_seed(seed: int, *parts: object) -> int:
    payload = ":".join((str(seed), *(str(part) for part in parts))).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=4).digest(), "big")


def _subset(project: LabeledProject, indices: np.ndarray) -> LabeledProject:
    selected = np.asarray(indices, dtype=np.int64)
    features = ProjectFeatures(
        family=project.features.family,
        name=project.features.name,
        values=freeze_array(project.features.values[selected], np.float64),
        feature_names=project.features.feature_names,
        effort=freeze_array(project.features.effort[selected], np.float64),
        effort_name=project.features.effort_name,
        row_ids=freeze_array(project.features.row_ids[selected], np.int64),
        cleaning=dict(project.features.cleaning),
    )
    return LabeledProject(
        features=features,
        labels=freeze_array(project.labels[selected], np.int64),
    )


def _outer_fold(
    project: LabeledProject, folds: int, fold: int, seed: int
) -> tuple[LabeledProject, LabeledProject]:
    counts = np.bincount(project.labels, minlength=2)
    if int(np.min(counts)) < folds:
        raise ValueError(
            f"{project.key} has class counts {counts.tolist()}, insufficient for {folds}-fold CV"
        )
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    splits = tuple(splitter.split(project.features.values, project.labels))
    train_indices, test_indices = splits[fold]
    return _subset(project, train_indices), _subset(project, test_indices)


def _fit_validation_split(
    project: LabeledProject, fraction: float, seed: int
) -> tuple[LabeledProject, LabeledProject | None]:
    counts = np.bincount(project.labels, minlength=2)
    if int(np.min(counts)) < 2:
        return project, None
    validation_rows = max(2, int(round(len(project.labels) * fraction)))
    validation_rows = min(validation_rows, len(project.labels) - 2)
    if validation_rows < 2:
        return project, None
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=validation_rows, random_state=seed
    )
    fit_indices, validation_indices = next(
        splitter.split(project.features.values, project.labels)
    )
    fit_project = _subset(project, fit_indices)
    validation_project = _subset(project, validation_indices)
    if len(np.unique(fit_project.labels)) < 2:
        return project, None
    return fit_project, validation_project


def _split_fit_validation(
    projects: Sequence[LabeledProject], fraction: float, seed: int
) -> tuple[Tuple[LabeledProject, ...], Tuple[LabeledProject, ...]]:
    fit_projects = []
    validation_projects = []
    for project in projects:
        fit_project, validation_project = _fit_validation_split(
            project, fraction, _derived_seed(seed, project.key, "validation")
        )
        fit_projects.append(fit_project)
        if validation_project is not None:
            validation_projects.append(validation_project)
    if not validation_projects:
        raise ValueError("No source project can provide a leakage-free validation split")
    validation_labels = np.concatenate(
        [project.labels for project in validation_projects]
    )
    if len(np.unique(validation_labels)) < 2:
        raise ValueError("Combined source validation data must contain both classes")
    return tuple(fit_projects), tuple(validation_projects)


def _evaluate(
    classifier: FittedValidationClassifier,
    projects: Sequence[LabeledProject],
) -> tuple[Dict[str, float], Dict[str, Mapping[str, float]]]:
    all_labels = []
    all_scores = []
    all_effort = []
    per_project: Dict[str, Mapping[str, float]] = {}
    for project in projects:
        scores = classifier.scores(project.features.values)
        per_project[project.key] = evaluate_predictions(
            project.labels,
            scores,
            project.features.effort,
            classifier.threshold,
        )
        all_labels.append(project.labels)
        all_scores.append(scores)
        all_effort.append(project.features.effort)
    aggregate = evaluate_predictions(
        np.concatenate(all_labels),
        np.concatenate(all_scores),
        np.concatenate(all_effort),
        classifier.threshold,
    )
    return aggregate, per_project


def _evaluate_formal_target(
    train_projects: Sequence[LabeledProject],
    target: LabeledProject,
    profile: ValidationProfile,
    seed: int,
) -> Dict[str, Any]:
    prediction = predict_validation_target(
        train_projects,
        target,
        profile.classifier,
        seed,
    )
    if prediction.evaluated_row_ids is not None and not np.array_equal(
        prediction.evaluated_row_ids,
        target.features.row_ids,
    ):
        raise ValueError(
            f"Downstream predictor changed target row order for {target.key}"
        )
    metrics = evaluate_predictions(
        target.labels,
        prediction.scores,
        target.features.effort,
        prediction.threshold,
    )
    return {
        "threshold": float(prediction.threshold),
        "metrics": metrics,
        "per_project_metrics": {target.key: metrics},
        "classifier": dict(prediction.metadata),
    }


def _quality_records(
    fit_projects: Sequence[LabeledProject],
    heldout_by_key: Mapping[str, LabeledProject],
    augmented_projects: Sequence[LabeledProject],
    profile: ValidationProfile,
    seed: int,
    augmentation_metadata: Mapping[str, Any],
) -> Tuple[Mapping[str, Any], ...]:
    augmented_by_key = {project.key: project for project in augmented_projects}
    diagnostics = {
        str(item.get("project")): item
        for item in augmentation_metadata.get("projects", ())
        if isinstance(item, Mapping) and item.get("project") is not None
    }
    records = []
    for fit_project in fit_projects:
        if fit_project.key not in augmented_by_key:
            raise ValueError(
                f"Oversampler omitted source project {fit_project.key}"
            )
        heldout_project = heldout_by_key.get(fit_project.key)
        if heldout_project is None:
            continue
        result = evaluate_synthetic_quality(
            fit_project,
            heldout_project,
            augmented_by_key[fit_project.key],
            profile.quality,
            _derived_seed(seed, fit_project.key, "quality"),
        )
        records.append(
            {
                "project": fit_project.key,
                "metrics": result.metrics,
                "metadata": result.metadata,
                "generation_diagnostics": diagnostics.get(fit_project.key, {}),
            }
        )
    return tuple(records)


def _utility_payload(
    baseline: FittedValidationClassifier,
    augmented: FittedValidationClassifier,
    test_projects: Sequence[LabeledProject],
    augmentation_metadata: Mapping[str, Any],
    duration_seconds: float,
) -> Dict[str, Any]:
    baseline_metrics, baseline_projects = _evaluate(baseline, test_projects)
    augmented_metrics, augmented_projects = _evaluate(augmented, test_projects)
    return {
        "baseline": {
            "threshold": baseline.threshold,
            "metrics": baseline_metrics,
            "per_project_metrics": baseline_projects,
            "classifier": baseline.metadata,
        },
        "augmented": {
            "threshold": augmented.threshold,
            "metrics": augmented_metrics,
            "per_project_metrics": augmented_projects,
            "classifier": augmented.metadata,
        },
        "delta_metrics": {
            metric: float(augmented_metrics[metric] - baseline_metrics[metric])
            for metric in METRIC_NAMES
        },
        "augmentation_metadata": augmentation_metadata,
        "duration_seconds": float(duration_seconds),
    }


def _single_downstream_utility_payload(
    augmented: Mapping[str, Any],
    augmentation_metadata: Mapping[str, Any],
    duration_seconds: float,
) -> Dict[str, Any]:
    return {
        "paired_baseline": False,
        "augmented": dict(augmented),
        "augmentation_metadata": augmentation_metadata,
        "duration_seconds": float(duration_seconds),
    }


def run_source_cv_fold(
    family_projects: Sequence[LabeledProject],
    strategy: OversamplingStrategy,
    profile: ValidationProfile,
    seed: int,
    fold: int,
    *,
    split_seed: Optional[int] = None,
    oversampling_seed: Optional[int] = None,
    classifier_seed: Optional[int] = None,
    quality_seed: Optional[int] = None,
) -> ProtocolResult:
    if profile.classifier.kind != "logistic_regression":
        raise ValueError(
            "source_cv is unavailable for the transductive formal downstream; "
            "use the cpdp protocol"
        )
    split_seed = int(seed if split_seed is None else split_seed)
    oversampling_seed = int(seed if oversampling_seed is None else oversampling_seed)
    classifier_seed = int(seed if classifier_seed is None else classifier_seed)
    quality_seed = int(seed if quality_seed is None else quality_seed)
    started = time.perf_counter()
    outer_train = []
    outer_test = []
    for project in family_projects:
        train_project, test_project = _outer_fold(
            project,
            profile.source_cv.folds,
            fold,
            _derived_seed(split_seed, project.key, "outer"),
        )
        outer_train.append(train_project)
        outer_test.append(test_project)
    fit_projects, validation_projects = _split_fit_validation(
        outer_train, profile.source_cv.validation_fraction, split_seed
    )
    baseline = fit_validation_classifier(
        fit_projects, validation_projects, profile.classifier, classifier_seed
    )
    augmentation = strategy.fit_resample(fit_projects, oversampling_seed)
    augmented = fit_validation_classifier(
        augmentation.projects,
        validation_projects,
        profile.classifier,
        classifier_seed,
    )
    utility = _utility_payload(
        baseline,
        augmented,
        outer_test,
        augmentation.metadata,
        time.perf_counter() - started,
    )
    quality = _quality_records(
        fit_projects,
        {project.key: project for project in outer_test},
        augmentation.projects,
        profile,
        quality_seed,
        augmentation.metadata,
    )
    return ProtocolResult(utility=utility, quality=quality)


def run_cpdp_target(
    sources: Sequence[LabeledProject],
    target: LabeledProject,
    strategy: OversamplingStrategy,
    profile: ValidationProfile,
    seed: int,
) -> ProtocolResult:
    started = time.perf_counter()
    if profile.classifier.kind == "formal_kan_jepa_tabr":
        augmentation = strategy.fit_resample(sources, seed)
        augmented = _evaluate_formal_target(
            augmentation.projects,
            target,
            profile,
            seed,
        )
        return ProtocolResult(
            utility=_single_downstream_utility_payload(
                augmented,
                augmentation.metadata,
                time.perf_counter() - started,
            ),
            quality=(),
        )

    fit_projects, validation_projects = _split_fit_validation(
        sources, profile.cpdp.validation_fraction, seed
    )
    baseline = fit_validation_classifier(
        fit_projects, validation_projects, profile.classifier, seed
    )
    augmentation = strategy.fit_resample(fit_projects, seed)
    augmented = fit_validation_classifier(
        augmentation.projects, validation_projects, profile.classifier, seed
    )
    utility = _utility_payload(
        baseline,
        augmented,
        (target,),
        augmentation.metadata,
        time.perf_counter() - started,
    )
    quality = _quality_records(
        fit_projects,
        {project.key: project for project in validation_projects},
        augmentation.projects,
        profile,
        seed,
        augmentation.metadata,
    )
    return ProtocolResult(utility=utility, quality=quality)
