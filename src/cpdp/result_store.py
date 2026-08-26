from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from cpdp.metrics import METRIC_NAMES


class ResultStore:
    def __init__(self, root: Path, save_predictions: bool) -> None:
        self.root = root
        self.save_predictions = save_predictions

    def run_path(
        self, profile: str, method: str, family: str, target: str, seed: int
    ) -> Path:
        return self.root / "runs" / profile / method / family / target / f"seed_{seed}.json"

    def prediction_path(
        self, profile: str, method: str, family: str, target: str, seed: int
    ) -> Path:
        return (
            self.root
            / "predictions"
            / profile
            / method
            / family
            / target
            / f"seed_{seed}.npz"
        )

    def is_complete(
        self,
        profile: str,
        method: str,
        family: str,
        target: str,
        seed: int,
        *,
        result_signature: Optional[str] = None,
    ) -> bool:
        path = self.run_path(profile, method, family, target, seed)
        if not path.is_file():
            return False
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if payload.get("status") != "success":
                return False
            if result_signature is None:
                return True
            return (
                payload.get("method_params", {}).get("result_signature")
                == result_signature
            )
        except (OSError, json.JSONDecodeError):
            return False

    @staticmethod
    def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(temporary, path)

    @staticmethod
    def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        frame.to_csv(temporary, index=False, encoding="utf-8-sig")
        os.replace(temporary, path)

    @staticmethod
    def _safe_component(value: str) -> str:
        if value in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
            raise ValueError(f"Unsafe summary path component: {value!r}")
        return value

    def _write_summary_set(
        self,
        directory: Path,
        runs: pd.DataFrame,
        target_means: pd.DataFrame,
        family_means: pd.DataFrame,
        overall_means: pd.DataFrame,
        prefix: str = "",
    ) -> Dict[str, Path]:
        paths = {
            "runs": directory / f"{prefix}run_results.csv",
            "targets": directory / f"{prefix}target_means.csv",
            "families": directory / f"{prefix}family_macro_means.csv",
            "overall": directory / f"{prefix}overall_project_macro_means.csv",
        }
        self._atomic_csv(paths["runs"], runs)
        self._atomic_csv(paths["targets"], target_means)
        self._atomic_csv(paths["families"], family_means)
        self._atomic_csv(paths["overall"], overall_means)
        return paths

    def save_success(
        self,
        payload: Dict[str, Any],
        labels: np.ndarray,
        scores: np.ndarray,
        effort: np.ndarray,
        row_ids: np.ndarray,
    ) -> None:
        method = str(payload["method"])
        family = str(payload["family"])
        target = str(payload["target"])
        profile = str(payload["profile"])
        seed = int(payload["seed"])
        if self.save_predictions:
            path = self.prediction_path(profile, method, family, target, seed)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp.npz")
            np.savez_compressed(
                temporary,
                labels=np.asarray(labels),
                scores=np.asarray(scores),
                effort=np.asarray(effort),
                row_ids=np.asarray(row_ids),
            )
            os.replace(temporary, path)
            payload["prediction_file"] = str(path)
        self._atomic_json(self.run_path(profile, method, family, target, seed), payload)

    def save_failure(self, payload: Dict[str, Any]) -> None:
        self._atomic_json(
            self.run_path(
                str(payload["profile"]),
                str(payload["method"]),
                str(payload["family"]),
                str(payload["target"]),
                int(payload["seed"]),
            ),
            payload,
        )

    def _records(self) -> Iterable[Dict[str, Any]]:
        for path in self.root.glob("runs/*/*/*/*/seed_*.json"):
            try:
                with path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue
            if payload.get("status") != "success":
                continue
            record = {
                "profile": payload["profile"],
                "method": payload["method"],
                "family": payload["family"],
                "target": payload["target"],
                "seed": payload["seed"],
                "threshold": payload["threshold"],
                "duration_seconds": payload["duration_seconds"],
            }
            record.update(payload["metrics"])
            yield record

    def summarize(
        self,
        expected_repetitions: Optional[int] = None,
        expected_seeds: Optional[Sequence[int]] = None,
    ) -> Dict[str, Path]:
        records = list(self._records())
        if expected_seeds is not None:
            allowed_seeds = {int(seed) for seed in expected_seeds}
            if not allowed_seeds:
                raise ValueError("expected_seeds must not be empty")
            records = [
                record
                for record in records
                if int(record["seed"]) in allowed_seeds
            ]
        if not records:
            raise RuntimeError(f"No successful run results under {self.root}")
        summary_dir = self.root / "summary"
        summary_dir.mkdir(parents=True, exist_ok=True)

        runs = pd.DataFrame(records).sort_values(
            ["profile", "method", "family", "target", "seed"]
        )

        aggregations = {metric: ["mean", "std", "count"] for metric in METRIC_NAMES}
        target_means = runs.groupby(
            ["profile", "method", "family", "target"], as_index=False
        ).agg(aggregations)
        target_means.columns = [
            "_".join(part for part in column if part)
            if isinstance(column, tuple)
            else column
            for column in target_means.columns
        ]
        count_columns = [f"{metric}_count" for metric in METRIC_NAMES]
        if expected_repetitions is not None:
            target_means["expected_runs"] = int(expected_repetitions)
            target_means["complete"] = target_means[count_columns].eq(
                int(expected_repetitions)
            ).all(axis=1)
        mean_columns = [f"{metric}_mean" for metric in METRIC_NAMES]
        family_means = target_means.groupby(
            ["profile", "method", "family"], as_index=False
        )[mean_columns].mean()
        family_coverage = target_means.groupby(
            ["profile", "method", "family"], as_index=False
        ).agg(
            target_count=("target", "nunique"),
            min_run_count=("auc_count", "min"),
            max_run_count=("auc_count", "max"),
        )
        family_means = family_coverage.merge(
            family_means, on=["profile", "method", "family"], how="left"
        )
        overall_means = target_means.groupby(
            ["profile", "method"], as_index=False
        )[mean_columns].mean()
        overall_coverage = target_means.groupby(
            ["profile", "method"], as_index=False
        ).agg(
            family_count=("family", "nunique"),
            target_count=("target", "nunique"),
            min_run_count=("auc_count", "min"),
            max_run_count=("auc_count", "max"),
        )
        overall_means = overall_coverage.merge(
            overall_means, on=["profile", "method"], how="left"
        )
        latest_paths = self._write_summary_set(
            summary_dir, runs, target_means, family_means, overall_means
        )

        by_method_dir = summary_dir / "by_method"
        for profile, method in runs.loc[:, ["profile", "method"]].drop_duplicates().itertuples(
            index=False, name=None
        ):
            safe_profile = self._safe_component(str(profile))
            safe_method = self._safe_component(str(method))
            run_mask = (runs["profile"] == profile) & (runs["method"] == method)
            target_mask = (target_means["profile"] == profile) & (
                target_means["method"] == method
            )
            family_mask = (family_means["profile"] == profile) & (
                family_means["method"] == method
            )
            overall_mask = (overall_means["profile"] == profile) & (
                overall_means["method"] == method
            )
            self._write_summary_set(
                by_method_dir / safe_profile / safe_method,
                runs.loc[run_mask].reset_index(drop=True),
                target_means.loc[target_mask].reset_index(drop=True),
                family_means.loc[family_mask].reset_index(drop=True),
                overall_means.loc[overall_mask].reset_index(drop=True),
                prefix=f"{safe_method}_",
            )

        snapshot_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        history_dir = summary_dir / "history" / snapshot_name
        self._write_summary_set(
            history_dir, runs, target_means, family_means, overall_means
        )
        return {
            **latest_paths,
            "by_method": by_method_dir,
            "history_snapshot": history_dir,
        }
