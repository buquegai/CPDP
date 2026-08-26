from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import linprog
from scipy.stats import rankdata

from cpdp.methods.base import BaseMethod, MethodProvenance
from cpdp.types import LabeledProject, PredictionResult, UnlabeledProject


def _matlab_zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0, ddof=1)
    std = np.where(np.isfinite(std) & (std > 0), std, 1.0)
    return (values - mean) / std


def _spearman_distance(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = rankdata(left)
    right_rank = rankdata(right)
    if np.std(left_rank) == 0 or np.std(right_rank) == 0:
        return 0.0
    correlation = float(np.corrcoef(left_rank, right_rank)[0, 1])
    return 0.0 if not np.isfinite(correlation) else 1.0 - correlation


def _corr_distance(source_x: np.ndarray, source_y: np.ndarray, target_x: np.ndarray) -> float:
    distances = np.asarray(
        [_spearman_distance(source_x[:, index], source_y) for index in range(source_x.shape[1])]
    )
    top_count = min(3, source_x.shape[1])
    selected = np.argsort(-distances, kind="stable")[:top_count]
    source_selected = source_x[:, selected].T
    target_selected = target_x[:, selected].T

    source_relation = np.zeros(top_count, dtype=np.float64)
    target_relation = np.zeros(top_count, dtype=np.float64)
    for row in range(top_count):
        for other in range(top_count):
            if row != other:
                source_relation[row] = _spearman_distance(
                    source_selected[row], source_selected[other]
                )
                target_relation[row] = _spearman_distance(
                    target_selected[row], target_selected[other]
                )
    return float(np.linalg.norm(source_relation - target_relation))


def _mpos(values: np.ndarray, labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    positive = np.asarray(values[labels == 1], dtype=np.float64)
    negative_count = int(np.sum(labels == 0))
    generated_count = negative_count - len(positive)
    if generated_count <= 0:
        return values.copy(), labels.copy()
    if len(positive) == 0:
        raise ValueError("MPOS requires at least one defective source instance")

    rng = np.random.default_rng(seed)
    choices = rng.integers(0, len(positive), size=(generated_count, values.shape[1]))
    feature_indices = np.arange(values.shape[1])[None, :]
    generated = positive[choices, feature_indices]
    balanced_values = np.vstack([generated, values])
    balanced_labels = np.concatenate(
        [np.ones(generated_count, dtype=np.int64), labels.astype(np.int64)]
    )
    return balanced_values, balanced_labels


def _write_arff(path: Path, values: np.ndarray, labels: np.ndarray) -> None:
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("@relation dssdpp_cfs\n\n")
        for index in range(values.shape[1]):
            handle.write(f"@attribute feature_{index + 1} numeric\n")
        handle.write("@attribute class {0,1}\n\n@data\n")
        for row, label in zip(values, labels):
            serialized = ",".join(format(float(value), ".17g") for value in row)
            handle.write(f"{serialized},{int(label)}\n")


def _weka_cfs_indices(values: np.ndarray, labels: np.ndarray, weka_jar: Path) -> np.ndarray:
    if not weka_jar.is_file():
        raise FileNotFoundError(f"Official Weka JAR not found: {weka_jar}")
    with tempfile.TemporaryDirectory(prefix="cpdp_dssdpp_") as temporary_directory:
        arff_path = Path(temporary_directory) / "source.arff"
        _write_arff(arff_path, values, labels)
        command = [
            "java",
            "-cp",
            str(weka_jar),
            "weka.attributeSelection.AttributeSelection",
            "weka.attributeSelection.CfsSubsetEval",
            "-s",
            "weka.attributeSelection.BestFirst -D 1 -N 5",
            "-i",
            str(arff_path),
            "-c",
            "last",
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    output = f"{completed.stdout}\n{completed.stderr}"
    if completed.returncode != 0:
        raise RuntimeError(f"Weka CFS failed with code {completed.returncode}:\n{output}")
    match = re.search(r"Selected attributes:\s*([0-9,]+)\s*:", output)
    if not match:
        raise RuntimeError(f"Could not parse Weka CFS output:\n{output}")
    selected = [int(value) - 1 for value in match.group(1).split(",")]
    selected = [index for index in selected if 0 <= index < values.shape[1]]
    if not selected:
        raise RuntimeError("Weka CFS selected no predictive features")
    return np.asarray(selected, dtype=np.int64)


def _domain_programming_scores(
    source_x: np.ndarray, source_y: np.ndarray, target_x: np.ndarray
) -> np.ndarray:
    classes = np.unique(source_y)
    if not np.array_equal(classes, np.array([0, 1])):
        raise ValueError("DSSDPP requires source labels containing both 0 and 1")

    target_count = target_x.shape[0]
    class_count = len(classes)
    distances = np.empty((target_count, class_count), dtype=np.float64)
    target_scale = np.std(target_x, axis=0, ddof=1)
    target_scale = np.where(np.isfinite(target_scale) & (target_scale > 0), target_scale, 1.0)
    for class_index, class_label in enumerate(classes):
        center = np.mean(source_x[source_y == class_label], axis=0)
        distances[:, class_index] = np.sqrt(
            np.sum(((target_x - center) / target_scale) ** 2, axis=1)
        )

    variable_count = target_count * class_count
    objective = distances.reshape(-1)
    equality = np.zeros((target_count, variable_count), dtype=np.float64)
    for target_index in range(target_count):
        start = target_index * class_count
        equality[target_index, start : start + class_count] = 1.0
    inequality = np.zeros((class_count, variable_count), dtype=np.float64)
    for class_index in range(class_count):
        inequality[class_index, class_index::class_count] = -1.0

    solution = linprog(
        objective,
        A_ub=inequality,
        b_ub=-np.ones(class_count),
        A_eq=equality,
        b_eq=np.ones(target_count),
        bounds=(0.0, 1.0),
        method="highs",
    )
    if not solution.success:
        raise RuntimeError(f"DSSDPP linear program failed: {solution.message}")
    membership = solution.x.reshape(target_count, class_count)
    return membership[:, int(np.where(classes == 1)[0][0])]


class DSSDPPMethod(BaseMethod):
    name = "dssdpp"
    provenance = MethodProvenance(
        display_name="DSSDPP",
        implementation="Python port of the official MATLAB implementation",
        fidelity="official-equivalent-port",
        source_url="https://zenodo.org/records/6651565",
        source_revision="DSSDPP_v3.zip md5:6672334b7a5e956f17f1ac4c7f65498c",
        target_label_policy="No target labels are passed to source selection or prediction",
        notes=(
            "The outer LOPO pool exposes every same-family source project. The official CORR "
            "selector remains an algorithm-internal source-selection step. Full target features "
            "replace the paper's random 10% unlabeled target subset under the unified protocol."
        ),
    )

    def fit_predict(
        self,
        sources: Sequence[LabeledProject],
        target: UnlabeledProject,
        seed: int,
    ) -> PredictionResult:
        if not sources:
            raise ValueError("DSSDPP requires at least one source project")
        target_x = _matlab_zscore(target.features.values)
        source_selection = str(self.params.get("source_selection", "corr")).lower()
        if source_selection != "corr":
            raise ValueError("The official DSSDPP demo uses CORR source selection")

        source_distances = []
        for source in sources:
            source_x = _matlab_zscore(source.features.values)
            source_distances.append(_corr_distance(source_x, source.labels, target_x))
        selected_index = int(np.argmin(source_distances))
        selected_source = sources[selected_index]

        balanced_x, balanced_y = _mpos(
            selected_source.features.values.copy(), selected_source.labels.copy(), seed
        )
        if bool(self.params.get("use_weka_cfs", True)):
            weka_jars = tuple(
                (
                    self.project_root
                    / "third_party"
                    / "original"
                    / "DSSDPP_v3"
                    / "DSSDPP_v3"
                    / "featureselection"
                    / "fspackage"
                    / "lib"
                    / "weka"
                ).glob("weka.jar")
            )
            if len(weka_jars) != 1:
                raise RuntimeError(f"Expected one official Weka JAR, found {len(weka_jars)}")
            weka_jar = weka_jars[0]
            feature_indices = _weka_cfs_indices(balanced_x, balanced_y, weka_jar)
        else:
            feature_indices = np.arange(balanced_x.shape[1], dtype=np.int64)

        source_selected = _matlab_zscore(balanced_x[:, feature_indices])
        target_selected = _matlab_zscore(target.features.values[:, feature_indices])
        scores = _domain_programming_scores(source_selected, balanced_y, target_selected)
        return PredictionResult(
            scores=scores,
            threshold=0.5,
            metadata={
                "selected_source": selected_source.key,
                "source_distances": {
                    source.key: float(distance)
                    for source, distance in zip(sources, source_distances)
                },
                "selected_feature_count": int(len(feature_indices)),
                "selected_features": [
                    target.features.feature_names[index] for index in feature_indices
                ],
                "imbalance_handling": "official MPOS",
            },
        )
