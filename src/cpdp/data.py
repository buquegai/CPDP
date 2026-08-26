from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np
import pandas as pd

from cpdp.config import DataConfig, FamilyConfig
from cpdp.types import LabeledProject, ProjectFeatures


@dataclass(frozen=True)
class CleaningReport:
    original_rows: int
    missing_rows_removed: int
    duplicate_rows_removed: int
    inconsistent_rows_removed: int
    final_rows: int

    def as_dict(self) -> Dict[str, int]:
        return {
            "original_rows": self.original_rows,
            "missing_rows_removed": self.missing_rows_removed,
            "duplicate_rows_removed": self.duplicate_rows_removed,
            "inconsistent_rows_removed": self.inconsistent_rows_removed,
            "final_rows": self.final_rows,
        }


def _clean_master_style(
    frame: pd.DataFrame, feature_columns: Tuple[str, ...], label_column: str
) -> Tuple[pd.DataFrame, np.ndarray, CleaningReport]:
    original_rows = len(frame)
    row_ids = np.arange(original_rows, dtype=np.int64)

    required = list(feature_columns) + [label_column]
    complete_mask = frame.loc[:, required].notna().all(axis=1).to_numpy()
    missing_rows_removed = int((~complete_mask).sum())
    cleaned = frame.loc[complete_mask, required].copy()
    kept_ids = row_ids[complete_mask]

    duplicate_mask = cleaned.duplicated(subset=required, keep="first").to_numpy()
    duplicate_rows_removed = int(duplicate_mask.sum())
    cleaned = cleaned.loc[~duplicate_mask].copy()
    kept_ids = kept_ids[~duplicate_mask]

    label_counts = cleaned.groupby(list(feature_columns), dropna=False)[label_column].transform(
        "nunique"
    )
    inconsistent_mask = label_counts.to_numpy() > 1
    inconsistent_rows_removed = int(inconsistent_mask.sum())
    cleaned = cleaned.loc[~inconsistent_mask].copy()
    kept_ids = kept_ids[~inconsistent_mask]

    report = CleaningReport(
        original_rows=original_rows,
        missing_rows_removed=missing_rows_removed,
        duplicate_rows_removed=duplicate_rows_removed,
        inconsistent_rows_removed=inconsistent_rows_removed,
        final_rows=len(cleaned),
    )
    return cleaned.reset_index(drop=True), kept_ids, report


def _freeze(array: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(array)
    contiguous.setflags(write=False)
    return contiguous


def load_project(data: DataConfig, family: FamilyConfig, project: str) -> LabeledProject:
    path = data.root / family.name / f"{project}{data.file_suffix}"
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    frame = pd.read_csv(path, encoding="utf-8-sig", na_values=["?"])
    if data.label_column not in frame.columns:
        raise ValueError(f"Missing label column {data.label_column!r} in {path}")
    if family.effort_column not in frame.columns:
        raise ValueError(f"Missing effort column {family.effort_column!r} in {path}")

    feature_columns = tuple(column for column in frame.columns if column != data.label_column)
    for column in feature_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame[data.label_column] = pd.to_numeric(frame[data.label_column], errors="coerce")

    if data.cleaning == "master":
        cleaned, row_ids, report = _clean_master_style(
            frame, feature_columns, data.label_column
        )
    elif data.cleaning == "none":
        cleaned = frame.loc[:, list(feature_columns) + [data.label_column]].copy()
        row_ids = np.arange(len(cleaned), dtype=np.int64)
        report = CleaningReport(len(cleaned), 0, 0, 0, len(cleaned))
    else:
        raise ValueError(f"Unknown cleaning policy: {data.cleaning}")

    values = cleaned.loc[:, feature_columns].to_numpy(dtype=np.float64)
    raw_labels = cleaned[data.label_column].to_numpy(dtype=np.float64)
    unique_labels = set(np.unique(raw_labels).tolist())
    if unique_labels != {0.0, 1.0}:
        raise ValueError(
            f"Label column must contain exactly binary values 0 and 1: {path}"
        )
    labels = raw_labels.astype(np.int64)
    effort = cleaned[family.effort_column].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite feature value remains after cleaning: {path}")
    if not np.isfinite(effort).all() or np.any(effort < 0):
        raise ValueError(f"Effort column must be finite and non-negative: {path}")
    features = ProjectFeatures(
        family=family.name,
        name=project,
        values=_freeze(values),
        feature_names=feature_columns,
        effort=_freeze(effort),
        effort_name=family.effort_column,
        row_ids=_freeze(row_ids),
        cleaning=report.as_dict(),
    )
    return LabeledProject(features=features, labels=_freeze(labels))


def load_all_projects(data: DataConfig) -> Mapping[str, LabeledProject]:
    projects: Dict[str, LabeledProject] = {}
    for family in data.families:
        expected_schema = None
        for project_name in family.projects:
            project = load_project(data, family, project_name)
            if expected_schema is None:
                expected_schema = project.features.feature_names
            elif project.features.feature_names != expected_schema:
                raise ValueError(
                    f"Feature schema mismatch inside {family.name}: {project.key}"
                )
            projects[project.key] = project
    return projects


def sources_for_target(
    projects: Mapping[str, LabeledProject], family: str, target: str
) -> Iterable[LabeledProject]:
    prefix = f"{family}/"
    target_key = f"{family}/{target}"
    return tuple(
        project
        for key, project in projects.items()
        if key.startswith(prefix) and key != target_key
    )
