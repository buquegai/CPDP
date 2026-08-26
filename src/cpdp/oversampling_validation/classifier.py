from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cpdp.oversampling_validation.config import ClassifierConfig
from cpdp.predictors.kan_jepa import KANJEPAPredictor
from cpdp.predictors.threshold import select_mcc_threshold
from cpdp.types import LabeledProject, PredictionResult


def _stack(projects: Sequence[LabeledProject]) -> tuple[np.ndarray, np.ndarray]:
    if not projects:
        raise ValueError("Classifier requires at least one project")
    return (
        np.vstack([project.features.values for project in projects]),
        np.concatenate([project.labels for project in projects]),
    )


@dataclass
class FittedValidationClassifier:
    model: Pipeline
    threshold: float
    metadata: dict

    def scores(self, values: np.ndarray) -> np.ndarray:
        probabilities = self.model.predict_proba(
            np.asarray(values, dtype=np.float64)
        )[:, 1]
        return np.asarray(probabilities, dtype=np.float64)


def fit_validation_classifier(
    train_projects: Sequence[LabeledProject],
    validation_projects: Sequence[LabeledProject],
    config: ClassifierConfig,
    seed: int,
) -> FittedValidationClassifier:
    if config.kind != "logistic_regression":
        raise ValueError("fit_validation_classifier requires logistic_regression")
    train_values, train_labels = _stack(train_projects)
    validation_values, validation_labels = _stack(validation_projects)
    if len(np.unique(train_labels)) != 2:
        raise ValueError("Classifier training data must contain both classes")
    model = Pipeline(
        steps=(
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=config.c,
                    max_iter=config.max_iter,
                    class_weight=config.class_weight,
                    solver="lbfgs",
                    random_state=int(seed),
                ),
            ),
        )
    )
    model.fit(train_values, train_labels)
    validation_scores = np.asarray(
        model.predict_proba(validation_values)[:, 1], dtype=np.float64
    )
    threshold = select_mcc_threshold(validation_labels, validation_scores)
    return FittedValidationClassifier(
        model=model,
        threshold=threshold,
        metadata={
            "configuration": asdict(config),
            "training_rows": int(len(train_labels)),
            "training_defects": int(np.sum(train_labels == 1)),
            "validation_rows": int(len(validation_labels)),
            "validation_defects": int(np.sum(validation_labels == 1)),
            "threshold_policy": "real-source-validation-mcc",
        },
    )


def predict_validation_target(
    train_projects: Sequence[LabeledProject],
    target: LabeledProject,
    config: ClassifierConfig,
    seed: int,
    validation_projects: Sequence[LabeledProject] = (),
) -> PredictionResult:
    """Predict one held-out project with the configured common downstream model."""
    if config.kind == "formal_kan_jepa_tabr":
        predictor = KANJEPAPredictor.from_mapping(config.predictor)
        prediction = predictor.fit_predict(
            train_projects,
            target.without_labels(),
            int(seed),
        )
        return PredictionResult(
            scores=np.asarray(prediction.scores, dtype=np.float64),
            threshold=float(prediction.threshold),
            metadata={
                "kind": config.kind,
                "method": config.method,
                "formal_predictor": dict(prediction.metadata),
            },
            evaluated_row_ids=prediction.evaluated_row_ids,
        )
    if config.kind == "logistic_regression":
        if not validation_projects:
            raise ValueError(
                "logistic_regression requires real-source validation projects"
            )
        fitted = fit_validation_classifier(
            train_projects,
            validation_projects,
            config,
            int(seed),
        )
        return PredictionResult(
            scores=fitted.scores(target.features.values),
            threshold=float(fitted.threshold),
            metadata=fitted.metadata,
            evaluated_row_ids=target.features.row_ids,
        )
    raise ValueError(f"Unsupported validation classifier: {config.kind}")
