from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

from cpdp.metrics import METRIC_NAMES
from cpdp.oversampling_validation.quality import QUALITY_METRICS


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


class ValidationStore:
    def __init__(self, root: Path, fingerprint: str, profile: str) -> None:
        self.root = root / fingerprint / profile
        self.fingerprint = fingerprint
        self.profile = profile

    def run_path(
        self,
        protocol: str,
        variant: str,
        family: str,
        seed: int,
        *,
        fold: Optional[int] = None,
        target: Optional[str] = None,
    ) -> Path:
        if protocol == "source_cv":
            if fold is None:
                raise ValueError("source_cv path requires fold")
            return (
                self.root
                / "runs"
                / protocol
                / variant
                / family
                / f"seed_{seed}"
                / f"fold_{fold}.json"
            )
        if protocol == "cpdp":
            if target is None:
                raise ValueError("cpdp path requires target")
            return (
                self.root
                / "runs"
                / protocol
                / variant
                / family
                / target
                / f"seed_{seed}.json"
            )
        raise ValueError(f"Unknown validation protocol: {protocol}")

    @staticmethod
    def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                _jsonable(payload),
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        os.replace(temporary, path)

    @staticmethod
    def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        frame.to_csv(temporary, index=False, encoding="utf-8-sig")
        os.replace(temporary, path)

    def is_complete(self, path: Path) -> bool:
        if not path.is_file():
            return False
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            return (
                payload.get("status") == "success"
                and payload.get("fingerprint") == self.fingerprint
            )
        except (OSError, json.JSONDecodeError):
            return False

    def save(self, path: Path, payload: Mapping[str, Any]) -> None:
        self._atomic_json(path, payload)

    def _records(self) -> Iterable[Mapping[str, Any]]:
        for path in self.root.glob("runs/**/*.json"):
            try:
                with path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue
            if payload.get("status") == "success":
                yield payload

    @staticmethod
    def _utility_row(payload: Mapping[str, Any]) -> Dict[str, Any]:
        utility = payload["utility"]
        augmentation = utility.get("augmentation_metadata", {})
        baseline = utility.get("baseline")
        augmented = utility["augmented"]
        row: Dict[str, Any] = {
            "fingerprint": payload["fingerprint"],
            "profile": payload["profile"],
            "protocol": payload["protocol"],
            "variant": payload["variant"],
            "family": payload["family"],
            "target": payload.get("target", "__source_cv__"),
            "seed": payload["seed"],
            "fold": payload.get("fold"),
            "duration_seconds": utility["duration_seconds"],
            "synthetic_samples": augmentation.get("total_synthetic_selected", 0),
            "baseline_threshold": None if baseline is None else baseline["threshold"],
            "augmented_threshold": augmented["threshold"],
        }
        for metric in METRIC_NAMES:
            row[f"baseline_{metric}"] = (
                None if baseline is None else baseline["metrics"][metric]
            )
            row[f"augmented_{metric}"] = augmented["metrics"][metric]
            row[f"delta_{metric}"] = utility.get("delta_metrics", {}).get(metric)
        return row

    @staticmethod
    def _quality_rows(payload: Mapping[str, Any]) -> Iterable[Dict[str, Any]]:
        augmentation = payload["utility"].get("augmentation_metadata", {})
        formula = augmentation.get("formula_bank", {})
        configuration = augmentation.get("configuration", {})
        formula_config = configuration.get("formula", {})
        diffusion_config = configuration.get("diffusion", {})
        for quality in payload.get("quality", ()):
            metadata = quality.get("metadata", {})
            diagnostics = quality.get("generation_diagnostics", {})
            numeric = diagnostics.get("numeric_validity", {})
            manifold = diagnostics.get("manifold", {})
            row: Dict[str, Any] = {
                "fingerprint": payload["fingerprint"],
                "profile": payload["profile"],
                "protocol": payload["protocol"],
                "variant": payload["variant"],
                "family": payload["family"],
                "target": payload.get("target", "__source_cv__"),
                "seed": payload["seed"],
                "fold": payload.get("fold"),
                "source_project": quality["project"],
                "quality_reason": metadata.get("reason"),
                "synthetic_samples": metadata.get("synthetic_samples"),
                "train_real_samples": metadata.get("train_real_samples"),
                "train_real_defects": metadata.get("train_real_defects"),
                "heldout_real_defects": metadata.get("heldout_real_defects"),
                "requested_synthetic": diagnostics.get("requested_synthetic"),
                "generated_candidates": diagnostics.get("generated_candidates"),
                "numerically_valid": diagnostics.get("numerically_valid"),
                "manifold_accepted": diagnostics.get("manifold_accepted"),
                "selected_synthetic": diagnostics.get("selected_synthetic"),
                "numeric_nonfinite_rows": numeric.get("nonfinite_rows"),
                "numeric_below_bound_rows": numeric.get("rows_below_lower_bound"),
                "numeric_above_bound_rows": numeric.get("rows_above_upper_bound"),
                "numeric_negative_rows": numeric.get(
                    "rows_with_negative_nonnegative_metric"
                ),
                "manifold_filter_enabled": manifold.get("enabled"),
                "manifold_filter_reason": manifold.get("reason"),
                "manifold_filter_lower": manifold.get("internal_distance_lower"),
                "manifold_filter_upper": manifold.get("internal_distance_upper"),
                "formula_enabled": formula.get("enabled"),
                "formula_reliability": formula.get("reliability"),
                "formula_validation_auc": formula.get("project_refit_held_out_auc"),
                "formula_reliability_weighting_enabled": formula_config.get(
                    "use_reliability_weighting"
                ),
                "formula_consistency_weight": diffusion_config.get(
                    "formula_consistency_weight"
                ),
                "formula_semantic_weight": diffusion_config.get(
                    "formula_semantic_weight"
                ),
                "project_support_projection": diffusion_config.get(
                    "project_support_projection"
                ),
            }
            for metric in QUALITY_METRICS:
                row[metric] = quality.get("metrics", {}).get(metric)
            yield row

    @staticmethod
    def _project_utility_rows(payload: Mapping[str, Any]) -> Iterable[Dict[str, Any]]:
        utility = payload["utility"]
        baseline = utility.get("baseline", {}).get("per_project_metrics", {})
        augmented = utility["augmented"]["per_project_metrics"]
        for project in sorted(augmented):
            row: Dict[str, Any] = {
                "fingerprint": payload["fingerprint"],
                "profile": payload["profile"],
                "protocol": payload["protocol"],
                "variant": payload["variant"],
                "family": payload["family"],
                "target": payload.get("target", "__source_cv__"),
                "evaluated_project": project,
                "seed": payload["seed"],
                "fold": payload.get("fold"),
            }
            for metric in METRIC_NAMES:
                baseline_value = baseline.get(project, {}).get(metric)
                row[f"baseline_{metric}"] = baseline_value
                row[f"augmented_{metric}"] = augmented[project][metric]
                row[f"delta_{metric}"] = (
                    None
                    if baseline_value is None
                    else augmented[project][metric] - baseline_value
                )
            yield row

    @staticmethod
    def _attach_none_baselines(
        frame: pd.DataFrame,
        *,
        project_level: bool = False,
    ) -> pd.DataFrame:
        if frame.empty or "variant" not in frame:
            return frame
        result = frame.copy()
        key_columns = ["protocol", "family", "target", "seed", "fold"]
        if project_level:
            key_columns.append("evaluated_project")
        pair_columns = []
        for column in key_columns:
            pair_column = f"__pair_{column}"
            result[pair_column] = result[column].astype("object").where(
                result[column].notna(), "__missing__"
            )
            pair_columns.append(pair_column)

        augmented_columns = [f"augmented_{metric}" for metric in METRIC_NAMES]
        if not project_level:
            augmented_columns.append("augmented_threshold")
        references = result.loc[
            result["variant"] == "none",
            pair_columns + augmented_columns,
        ].drop_duplicates(pair_columns, keep="last")
        references = references.rename(
            columns={column: f"__none_{column}" for column in augmented_columns}
        )
        result = result.merge(references, on=pair_columns, how="left", validate="many_to_one")

        for metric in METRIC_NAMES:
            baseline_column = f"baseline_{metric}"
            augmented_column = f"augmented_{metric}"
            reference_column = f"__none_{augmented_column}"
            result[baseline_column] = result[baseline_column].where(
                result[baseline_column].notna(), result[reference_column]
            )
            result[baseline_column] = pd.to_numeric(
                result[baseline_column], errors="coerce"
            )
            result[augmented_column] = pd.to_numeric(
                result[augmented_column], errors="coerce"
            )
            result[f"delta_{metric}"] = (
                result[augmented_column] - result[baseline_column]
            )
        if not project_level:
            result["baseline_threshold"] = result["baseline_threshold"].where(
                result["baseline_threshold"].notna(),
                result["__none_augmented_threshold"],
            )
            result["baseline_threshold"] = pd.to_numeric(
                result["baseline_threshold"], errors="coerce"
            )
        return result.drop(
            columns=pair_columns
            + [f"__none_{column}" for column in augmented_columns]
        )

    @staticmethod
    def _paired_rank_biserial(values: np.ndarray) -> Optional[float]:
        nonzero = np.asarray(values, dtype=np.float64)
        nonzero = nonzero[np.abs(nonzero) > 1e-15]
        if not len(nonzero):
            return 0.0
        ranks = rankdata(np.abs(nonzero))
        denominator = float(np.sum(ranks))
        if denominator <= 0:
            return None
        positive = float(np.sum(ranks[nonzero > 0]))
        negative = float(np.sum(ranks[nonzero < 0]))
        return (positive - negative) / denominator

    def summarize(self) -> Mapping[str, Path]:
        records = list(self._records())
        if not records:
            raise RuntimeError(f"No successful validation results under {self.root}")
        utility = pd.DataFrame([self._utility_row(record) for record in records])
        quality = pd.DataFrame(
            [row for record in records for row in self._quality_rows(record)]
        )
        project_utility = pd.DataFrame(
            [
                row
                for record in records
                for row in self._project_utility_rows(record)
            ]
        )
        utility = self._attach_none_baselines(utility)
        project_utility = self._attach_none_baselines(
            project_utility,
            project_level=True,
        )
        utility = utility.sort_values(
            ["protocol", "variant", "family", "target", "seed", "fold"],
            na_position="last",
        )
        if not quality.empty:
            quality = quality.sort_values(
                [
                    "protocol",
                    "variant",
                    "family",
                    "target",
                    "source_project",
                    "seed",
                    "fold",
                ],
                na_position="last",
            )
        if not project_utility.empty:
            project_utility = project_utility.sort_values(
                [
                    "protocol",
                    "variant",
                    "family",
                    "target",
                    "evaluated_project",
                    "seed",
                    "fold",
                ],
                na_position="last",
            )
        group_columns = ["protocol", "variant", "family", "target"]
        summary_rows = []
        for keys, frame in utility.groupby(group_columns, dropna=False):
            row = dict(zip(group_columns, keys))
            row["run_count"] = int(len(frame))
            row["seed_count"] = int(frame["seed"].nunique())
            row["mean_synthetic_samples"] = float(frame["synthetic_samples"].mean())
            per_seed = frame.groupby("seed", as_index=False).mean(numeric_only=True)
            for metric in METRIC_NAMES:
                values = per_seed[f"delta_{metric}"].dropna()
                row[f"mean_delta_{metric}"] = float(values.mean())
                row[f"median_delta_{metric}"] = float(values.median())
                row[f"std_delta_{metric}"] = float(values.std(ddof=1)) if len(values) > 1 else None
                row[f"win_rate_{metric}"] = float((values > 0).mean())
                paired_values = values.to_numpy(dtype=np.float64)
                if len(paired_values) < 2:
                    row[f"wilcoxon_p_{metric}"] = None
                elif np.all(np.abs(paired_values) <= 1e-15):
                    row[f"wilcoxon_p_{metric}"] = 1.0
                else:
                    row[f"wilcoxon_p_{metric}"] = float(
                        wilcoxon(paired_values, zero_method="wilcox").pvalue
                    )
                row[f"rank_biserial_{metric}"] = self._paired_rank_biserial(
                    paired_values
                )
            summary_rows.append(row)
        utility_summary = pd.DataFrame(summary_rows).sort_values(group_columns)

        summary_dir = self.root / "summary"
        paths = {
            "utility_runs": summary_dir / "utility_runs.csv",
            "quality_runs": summary_dir / "quality_runs.csv",
            "project_utility_runs": summary_dir / "project_utility_runs.csv",
            "utility_summary": summary_dir / "utility_summary.csv",
            "manifest": summary_dir / "manifest.json",
        }
        self._atomic_csv(paths["utility_runs"], utility)
        self._atomic_csv(paths["quality_runs"], quality)
        self._atomic_csv(paths["project_utility_runs"], project_utility)
        self._atomic_csv(paths["utility_summary"], utility_summary)
        self._atomic_json(
            paths["manifest"],
            {
                "fingerprint": self.fingerprint,
                "profile": self.profile,
                "successful_runs": len(records),
                "utility_rows": len(utility),
                "quality_rows": len(quality),
                "project_utility_rows": len(project_utility),
            },
        )
        return paths
