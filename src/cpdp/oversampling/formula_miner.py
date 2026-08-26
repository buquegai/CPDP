from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from cpdp.oversampling.legacy_config import FormulaConfig
from cpdp.oversampling.symbolic_library import FittedSymbolicLibrary
from cpdp.oversampling.types import FormulaBank, freeze_array
from cpdp.types import LabeledProject


def _fit_logistic(values: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(
        penalty="l2",
        solver="liblinear",
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        random_state=0,
    )
    model.fit(values, labels)
    return model


class SparseFormulaMiner:
    """Mine and freeze a sparse label-grounded symbolic formula bank."""

    def __init__(self, config: FormulaConfig) -> None:
        self.config = config

    def fit(
        self,
        projects: Sequence[LabeledProject],
        library: FittedSymbolicLibrary,
    ) -> FormulaBank:
        raw_values = np.vstack([project.features.values for project in projects])
        labels = np.concatenate([project.labels for project in projects])
        project_ids = np.concatenate(
            [np.full(project.features.n_samples, index) for index, project in enumerate(projects)]
        )
        symbolic_values = library.transform_numpy(raw_values)
        support = self._select_support(symbolic_values, labels)

        global_model = _fit_logistic(symbolic_values[:, support], labels)
        global_coefficients = np.zeros(symbolic_values.shape[1], dtype=np.float64)
        global_coefficients[support] = global_model.coef_[0]
        global_intercept = float(global_model.intercept_[0])

        validation_auc = self._project_held_out_auc(
            symbolic_values, labels, project_ids, len(projects)
        )
        reliability = float(np.clip(2.0 * (validation_auc - 0.5), 0.0, 1.0))
        enabled = bool(reliability >= self.config.minimum_reliability)

        project_coefficients = []
        project_intercepts = []
        for project_index, project in enumerate(projects):
            mask = project_ids == project_index
            project_labels = labels[mask]
            class_counts = np.bincount(project_labels, minlength=2)
            if int(class_counts.min()) < self.config.minimum_class_samples:
                project_coefficients.append(global_coefficients.copy())
                project_intercepts.append(global_intercept)
                continue
            local_model = _fit_logistic(symbolic_values[mask][:, support], project_labels)
            local_coefficients = np.zeros_like(global_coefficients)
            local_coefficients[support] = local_model.coef_[0]
            blend = len(project_labels) / (
                len(project_labels) + self.config.project_blend_strength
            )
            project_coefficients.append(
                (1.0 - blend) * global_coefficients + blend * local_coefficients
            )
            project_intercepts.append(
                (1.0 - blend) * global_intercept
                + blend * float(local_model.intercept_[0])
            )

        return FormulaBank(
            term_names=library.term_names,
            term_centers=freeze_array(library.centers, np.float64),
            term_scales=freeze_array(library.scales, np.float64),
            global_coefficients=freeze_array(global_coefficients, np.float64),
            global_intercept=global_intercept,
            project_coefficients=freeze_array(project_coefficients, np.float64),
            project_intercepts=freeze_array(project_intercepts, np.float64),
            project_names=tuple(project.key for project in projects),
            reliability=reliability,
            validation_auc=validation_auc,
            enabled=enabled,
        )

    def _select_support(self, values: np.ndarray, labels: np.ndarray) -> np.ndarray:
        selector = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=self.config.l1_c,
            class_weight="balanced",
            max_iter=2000,
            random_state=0,
        )
        selector.fit(values, labels)
        strengths = np.abs(selector.coef_[0])
        nonzero = np.flatnonzero(strengths > 1e-10)
        if len(nonzero) == 0:
            centered_labels = labels.astype(np.float64) - float(np.mean(labels))
            strengths = np.abs(values.T @ centered_labels)
            nonzero = np.flatnonzero(strengths > 0.0)
        if len(nonzero) == 0:
            nonzero = np.asarray([0], dtype=np.int64)
        order = nonzero[np.argsort(-strengths[nonzero])]
        return order[: self.config.max_terms]

    def _project_held_out_auc(
        self,
        values: np.ndarray,
        labels: np.ndarray,
        project_ids: np.ndarray,
        project_count: int,
    ) -> float:
        held_labels: List[np.ndarray] = []
        held_scores: List[np.ndarray] = []
        for project_index in range(project_count):
            test_mask = project_ids == project_index
            train_mask = ~test_mask
            if len(np.unique(labels[test_mask])) < 2 or len(np.unique(labels[train_mask])) < 2:
                continue
            fold_support = self._select_support(values[train_mask], labels[train_mask])
            model = _fit_logistic(
                values[train_mask][:, fold_support], labels[train_mask]
            )
            held_labels.append(labels[test_mask])
            held_scores.append(
                model.predict_proba(values[test_mask][:, fold_support])[:, 1]
            )
        if not held_labels:
            return 0.5
        return float(roc_auc_score(np.concatenate(held_labels), np.concatenate(held_scores)))


class FrozenFormulaGuidance:
    """Differentiable losses backed by coefficients frozen before DDPM training."""

    def __init__(
        self,
        library: FittedSymbolicLibrary,
        formula_bank: FormulaBank,
        use_reliability_weighting: bool = False,
    ) -> None:
        self.library = library
        self.formula_bank = formula_bank
        self.use_reliability_weighting = bool(use_reliability_weighting)

    def logits(self, raw_values, project_ids):
        import torch

        symbolic = self.library.transform_torch(raw_values)
        coefficients = torch.tensor(
            self.formula_bank.project_coefficients,
            dtype=raw_values.dtype,
            device=raw_values.device,
        )[project_ids]
        intercepts = torch.tensor(
            self.formula_bank.project_intercepts,
            dtype=raw_values.dtype,
            device=raw_values.device,
        )[project_ids]
        return torch.sum(symbolic * coefficients, dim=1) + intercepts

    def losses(
        self,
        predicted_raw,
        clean_raw,
        labels,
        project_ids,
        valid_mask,
    ):
        import torch
        import torch.nn.functional as functional

        if not self.formula_bank.enabled or not bool(torch.any(valid_mask)):
            zero = predicted_raw.sum() * 0.0
            return zero, zero
        predicted_logits = self.logits(predicted_raw[valid_mask], project_ids[valid_mask])
        with torch.no_grad():
            clean_logits = self.logits(clean_raw[valid_mask], project_ids[valid_mask])
        consistency = functional.smooth_l1_loss(
            torch.sigmoid(predicted_logits), torch.sigmoid(clean_logits)
        )
        semantic = functional.binary_cross_entropy_with_logits(
            predicted_logits, labels[valid_mask].to(predicted_logits.dtype)
        )
        if not self.use_reliability_weighting:
            return consistency, semantic
        reliability = self.formula_bank.reliability
        return consistency * reliability, semantic * reliability
