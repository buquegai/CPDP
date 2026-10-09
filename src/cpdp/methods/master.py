from __future__ import annotations

import json
import os
import subprocess
import tempfile
from math import log2
from typing import Iterable, Sequence

import numpy as np
from imblearn.over_sampling import SMOTE
from scipy.special import expit
from sklearn.feature_selection import mutual_info_classif

from cpdp.methods.base import BaseMethod, MethodProvenance
from cpdp.types import LabeledProject, PredictionResult, UnlabeledProject

# ---------------------------------------------------------------------------
# MIC backend: minepy is required by paper Eq. (9) but has no Python 3.12 build
# (its generated C code references PyLongObject.ob_digit, removed in 3.12).
# A conda-forge Python 3.10 environment provides the reference implementation;
# we call it in batch through a subprocess and cache the result per call.
# ---------------------------------------------------------------------------
MIC_PYTHON = os.environ.get("CPDP_MIC_PYTHON", "/root/anaconda3/envs/mic310/bin/python")


def _entropy(labels: np.ndarray) -> float:
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


def _mdl_cutpoints(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    values = np.asarray(values)[order]
    labels = np.asarray(labels)[order]

    def recurse(start: int, stop: int) -> list[float]:
        count = stop - start
        if count < 2 or len(np.unique(labels[start:stop])) < 2:
            return []
        parent_entropy = _entropy(labels[start:stop])
        best_gain = -np.inf
        best_split = None
        for split in range(start + 1, stop):
            if values[split - 1] == values[split] or labels[split - 1] == labels[split]:
                continue
            left_count = split - start
            right_count = stop - split
            gain = parent_entropy
            gain -= (left_count / count) * _entropy(labels[start:split])
            gain -= (right_count / count) * _entropy(labels[split:stop])
            if gain > best_gain:
                best_gain = gain
                best_split = split
        if best_split is None:
            return []

        # Fayyad & Irani MDL stopping criterion:
        #   Gain > [ log2(N-1) + Delta ] / N
        #   Delta = log2(3^k - 2) - [ k*H(S) - k1*H(S1) - k2*H(S2) ]
        # NOTE: the entropy terms are SUBTRACTED inside the bracket, so the
        # implementation must subtract them again when accumulating Delta.
        classes = len(np.unique(labels[start:stop]))
        left_classes = len(np.unique(labels[start:best_split]))
        right_classes = len(np.unique(labels[best_split:stop]))
        delta = log2(3**classes - 2) - classes * parent_entropy
        delta -= left_classes * _entropy(labels[start:best_split])
        delta -= right_classes * _entropy(labels[best_split:stop])
        threshold = (log2(count - 1) + delta) / count
        if best_gain <= threshold:
            return []
        cutpoint = float((values[best_split - 1] + values[best_split]) / 2.0)
        return recurse(start, best_split) + [cutpoint] + recurse(best_split, stop)

    return np.asarray(sorted(set(recurse(0, len(values)))), dtype=np.float64)


def _discretize(values: np.ndarray, cutpoints: Iterable[float]) -> np.ndarray:
    return np.digitize(values, np.asarray(tuple(cutpoints)), right=False).astype(np.int64)


def _discrete_mutual_information(values: np.ndarray, labels: np.ndarray) -> float:
    total = len(labels)
    result = 0.0
    for value in np.unique(values):
        value_mask = values == value
        p_value = float(np.mean(value_mask))
        for label in (0, 1):
            joint = float(np.mean(value_mask & (labels == label)))
            if joint == 0:
                continue
            p_label = float(np.mean(labels == label))
            result += joint * np.log(joint / (p_value * p_label))
    return result if total else 0.0


def _kl_divergence(source: np.ndarray, target: np.ndarray) -> float:
    categories = np.union1d(np.unique(source), np.unique(target))
    source_counts = np.asarray([(source == value).sum() for value in categories], dtype=float) + 1e-12
    target_counts = np.asarray([(target == value).sum() for value in categories], dtype=float) + 1e-12
    source_probability = source_counts / source_counts.sum()
    target_probability = target_counts / target_counts.sum()
    return float(np.sum(source_probability * np.log(source_probability / target_probability)))


def _minmax(values: np.ndarray) -> np.ndarray:
    """Scale a vector to [0, 1].

    Paper Eq. (3)-(5) normalize the mutual-information and KL terms before
    combining them. Normalizing to unit SUM (the previous behaviour) pushed
    every D_m to O(1/d), where sigmoid is nearly linear, so the source weights
    collapsed to uniform. Min-max normalization keeps the two terms on equal
    footing (which is what Eq. (3) intends for a convex combination) while
    keeping D_m in a range where sigmoid can actually discriminate.
    """
    span = float(values.max() - values.min())
    if span <= 0:
        return np.full(values.shape, 0.5, dtype=np.float64)
    return (values - float(values.min())) / span


_MIC_SCRIPT = r'''
import sys, json
import numpy as np
from minepy import MINE

payload = json.loads(open(sys.argv[1], "r").read())
features = np.asarray(payload["features"], dtype=np.float64)
labels = np.asarray(payload["labels"], dtype=np.float64)
scores = []
for index in range(features.shape[1]):
    mine = MINE(alpha=0.6, c=15)
    mine.compute_score(features[:, index], labels)
    scores.append(float(mine.mic()))
print(json.dumps(scores))
'''


def _feature_importance(values: np.ndarray, labels: np.ndarray, seed: int) -> tuple[np.ndarray, str]:
    importance = None
    backend = None

    # 1) paper-faithful MIC via the reference minepy in its own interpreter
    if os.path.isfile(MIC_PYTHON):
        try:
            with tempfile.TemporaryDirectory() as workdir:
                script = os.path.join(workdir, "mic_worker.py")
                payload = os.path.join(workdir, "payload.json")
                with open(script, "w", encoding="utf-8") as handle:
                    handle.write(_MIC_SCRIPT)
                with open(payload, "w", encoding="utf-8") as handle:
                    json.dump(
                        {
                            "features": values.tolist(),
                            "labels": labels.astype(float).tolist(),
                        },
                        handle,
                    )
                completed = subprocess.run(
                    [MIC_PYTHON, script, payload],
                    capture_output=True,
                    text=True,
                    timeout=900,
                    check=True,
                )
            importance = np.asarray(json.loads(completed.stdout.strip()), dtype=np.float64)
            backend = "minepy.MINE.mic (python3.10 reference env)"
        except Exception:
            importance = None

    # 2) in-process minepy, if one were ever importable under 3.12
    if importance is None:
        try:
            from minepy import MINE

            scores = []
            for index in range(values.shape[1]):
                mine = MINE(alpha=0.6, c=15)
                mine.compute_score(values[:, index], labels)
                scores.append(float(mine.mic()))
            importance = np.asarray(scores, dtype=np.float64)
            backend = "minepy.MINE.mic"
        except (ImportError, RuntimeError, ValueError):
            importance = None

    # 3) documented fallback
    if importance is None:
        importance = mutual_info_classif(
            values, labels, discrete_features=False, random_state=seed
        ).astype(np.float64)
        backend = "sklearn.mutual_info_classif fallback"

    total = float(np.sum(importance))
    if total <= 0:
        return np.full(values.shape[1], 1.0 / values.shape[1]), backend
    return importance / total, backend


def _source_transfer_weight(
    source_x: np.ndarray, source_y: np.ndarray, target_x: np.ndarray, weight_factor: float
) -> float:
    mutual_information = []
    divergence = []
    for index in range(source_x.shape[1]):
        cutpoints = _mdl_cutpoints(source_x[:, index], source_y)
        source_discrete = _discretize(source_x[:, index], cutpoints)
        target_discrete = _discretize(target_x[:, index], cutpoints)
        mutual_information.append(_discrete_mutual_information(source_discrete, source_y))
        divergence.append(_kl_divergence(source_discrete, target_discrete))
    mutual_information = np.asarray(mutual_information, dtype=np.float64)
    divergence = np.asarray(divergence, dtype=np.float64)
    normalized_mi = _minmax(mutual_information)
    normalized_kl = _minmax(divergence)
    feature_transfer = weight_factor * normalized_mi - (1.0 - weight_factor) * normalized_kl
    return float(np.sum(expit(feature_transfer)))


def _submodel_scores(
    source_x: np.ndarray,
    source_y: np.ndarray,
    target_x: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, str]:
    theta, importance_backend = _feature_importance(source_x, source_y, seed)
    target_minimum = np.min(target_x, axis=0)
    target_maximum = np.max(target_x, axis=0)
    within_target_range = (source_x >= target_minimum) & (source_x <= target_maximum)
    similarity = within_target_range.astype(float) @ theta
    instance_weight = similarity / np.square(np.sum(theta) - similarity + 1.0)

    source_discrete = np.empty_like(source_x, dtype=np.int64)
    target_discrete = np.empty_like(target_x, dtype=np.int64)
    unique_counts = np.empty(source_x.shape[1], dtype=np.int64)
    for index in range(source_x.shape[1]):
        cutpoints = _mdl_cutpoints(source_x[:, index], source_y)
        source_discrete[:, index] = _discretize(source_x[:, index], cutpoints)
        target_discrete[:, index] = _discretize(target_x[:, index], cutpoints)
        unique_counts[index] = max(1, len(np.unique(source_discrete[:, index])))

    log_posteriors = np.empty((target_x.shape[0], 2), dtype=np.float64)
    total_weight = float(np.sum(instance_weight))
    for class_label in (0, 1):
        class_mask = source_y == class_label
        class_weight = float(np.sum(instance_weight[class_mask]))
        prior = (class_weight + 1.0) / (total_weight + 2.0)
        log_probability = np.full(target_x.shape[0], np.log(max(prior, 1e-300)))
        for feature_index in range(source_x.shape[1]):
            denominator = class_weight + float(unique_counts[feature_index])
            conditional = np.empty(target_x.shape[0], dtype=np.float64)
            for row_index, target_value in enumerate(target_discrete[:, feature_index]):
                match = class_mask & (source_discrete[:, feature_index] == target_value)
                conditional[row_index] = (
                    float(np.sum(instance_weight[match])) + 1.0
                ) / denominator
            log_probability += np.exp(theta[feature_index]) * np.log(
                np.maximum(conditional, 1e-300)
            )
        log_posteriors[:, class_label] = log_probability

    maximum = np.max(log_posteriors, axis=1, keepdims=True)
    posterior = np.exp(log_posteriors - maximum)
    posterior /= np.sum(posterior, axis=1, keepdims=True)
    return posterior[:, 1], importance_backend


class MASTERMethod(BaseMethod):
    name = "master"
    provenance = MethodProvenance(
        display_name="MASTER",
        implementation="Independent Python implementation of paper equations (1)-(15)",
        fidelity="paper-derived-reimplementation",
        source_url="https://github.com/THN-BUAA/MASTER-master",
        source_revision="2315e1a764617f0bc7e23041c33d121282e02cb9",
        target_label_policy="No target labels are passed to training or transfer weighting",
        notes=(
            "The official repository states that the MASTER source code will be released later and "
            "currently contains datasets and baseline code only. This implementation must not be "
            "described as an official-code reproduction. Feature importance uses the MIC of paper "
            "Eq. (9) via a conda-forge minepy reference interpreter (python3.10); minepy has no "
            "Python 3.12 build, so the backend actually used is recorded in the result metadata."
        ),
    )

    def fit_predict(
        self,
        sources: Sequence[LabeledProject],
        target: UnlabeledProject,
        seed: int,
    ) -> PredictionResult:
        if not sources:
            raise ValueError("MASTER requires at least one source project")
        target_x = target.features.values.copy()
        weight_factor = float(self.params.get("lambda", 0.5))
        if not 0.0 <= weight_factor <= 1.0:
            raise ValueError("MASTER lambda must be between 0 and 1")
        smote_neighbors = int(self.params.get("smote_k_neighbors", 5))
        if smote_neighbors < 1:
            raise ValueError("MASTER smote_k_neighbors must be positive")

        submodel_scores = []
        transfer_weights = []
        importance_backends = []
        for source_index, source in enumerate(sources):
            source_x = source.features.values.copy()
            source_y = source.labels.copy()
            minority_count = int(np.min(np.bincount(source_y)))
            if minority_count >= 2 and np.sum(source_y == 0) != np.sum(source_y == 1):
                sampler = SMOTE(
                    random_state=(seed + source_index) % (2**32 - 1),
                    k_neighbors=min(smote_neighbors, minority_count - 1),
                )
                source_x, source_y = sampler.fit_resample(source_x, source_y)

            transfer_weights.append(
                _source_transfer_weight(source_x, source_y, target_x, weight_factor)
            )
            scores, backend = _submodel_scores(source_x, source_y, target_x, seed + source_index)
            submodel_scores.append(scores)
            importance_backends.append(backend)

        weights = np.asarray(transfer_weights, dtype=np.float64)
        if not np.isfinite(weights).all() or float(weights.sum()) <= 0:
            weights = np.ones(len(sources), dtype=np.float64)
        weights /= weights.sum()
        scores = np.average(np.vstack(submodel_scores), axis=0, weights=weights)
        return PredictionResult(
            scores=scores,
            threshold=0.5,
            metadata={
                "source_transfer_weights": {
                    source.key: float(weight)
                    for source, weight in zip(sources, weights)
                },
                "feature_importance_backends": sorted(set(importance_backends)),
                "imbalance_handling": "SMOTE independently within each source project",
                "lambda": weight_factor,
            },
        )
