from __future__ import annotations

import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

from cpdp.data import load_all_projects, sources_for_target
from cpdp.oversampling_validation.config import OversamplingValidationConfig
from cpdp.oversampling_validation.fingerprint import experiment_fingerprint
from cpdp.oversampling_validation.protocols import (
    run_cpdp_target,
    run_source_cv_fold,
)
from cpdp.oversampling_validation.store import ValidationStore
from cpdp.oversampling_validation.strategies import create_strategy
from cpdp.randomness import seed_everything


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ValidationTask:
    protocol: str
    variant: str
    family: str
    seed: int
    fold: Optional[int] = None
    target: Optional[str] = None


@dataclass(frozen=True)
class ValidationRunReport:
    fingerprint: str
    selected: int
    attempted: int
    succeeded: int
    failed: int
    skipped: int
    output_root: Path


class OversamplingValidationRunner:
    def __init__(self, config: OversamplingValidationConfig) -> None:
        self.config = config

    def _base_oversampling_config(self, profile_name: str) -> Mapping[str, Any]:
        profile = self.config.profile(profile_name)
        params = self.config.experiment.method_params(
            profile.oversampling_profile, self.config.oversampling_method
        )
        oversampling = params.get("oversampling")
        if not isinstance(oversampling, Mapping):
            raise ValueError(
                f"{profile.oversampling_profile}/{self.config.oversampling_method} "
                "must define an oversampling mapping"
            )
        return oversampling

    def fingerprint(self, profile_name: str) -> str:
        base = self._base_oversampling_config(profile_name)
        return experiment_fingerprint(self.config, profile_name, base)

    def _selected_variants(
        self, profile_name: str, requested: Optional[Sequence[str]]
    ) -> Tuple[str, ...]:
        configured = self.config.profile(profile_name).variants
        if not requested or tuple(requested) == ("all",):
            return configured
        selected = tuple(dict.fromkeys(str(name) for name in requested))
        unknown = sorted(set(selected) - set(configured))
        if unknown:
            raise KeyError(
                f"Variants are not enabled in profile {profile_name}: {', '.join(unknown)}"
            )
        return selected

    def _selected_seeds(
        self, profile_name: str, requested: Optional[Sequence[int]]
    ) -> Tuple[int, ...]:
        configured = self.config.profile(profile_name).seeds
        if not requested:
            return configured
        selected = tuple(dict.fromkeys(int(seed) for seed in requested))
        unknown = sorted(set(selected) - set(configured))
        if unknown:
            raise ValueError(
                f"Seeds are not enabled in profile {profile_name}: {unknown}"
            )
        return selected

    def _selected_folds(
        self, profile_name: str, requested: Optional[Sequence[int]]
    ) -> Tuple[int, ...]:
        configured = tuple(range(self.config.profile(profile_name).source_cv.folds))
        if not requested:
            return configured
        selected = tuple(dict.fromkeys(int(fold) for fold in requested))
        unknown = sorted(set(selected) - set(configured))
        if unknown:
            raise ValueError(
                f"Folds are not enabled in profile {profile_name}: {unknown}"
            )
        return selected

    def _selected_families(
        self, requested: Optional[Sequence[str]]
    ) -> Tuple[str, ...]:
        configured = tuple(family.name for family in self.config.experiment.data.families)
        if not requested:
            return configured
        lookup = {name.lower(): name for name in configured}
        selected = []
        for value in requested:
            try:
                selected.append(lookup[value.lower()])
            except KeyError as exc:
                raise KeyError(f"Unknown dataset family: {value}") from exc
        return tuple(dict.fromkeys(selected))

    def _selected_targets(
        self,
        families: Sequence[str],
        requested: Optional[Sequence[str]],
    ) -> Tuple[Tuple[str, str], ...]:
        configured = tuple(
            (family, target)
            for family, target in self.config.experiment.targets()
            if family in families
        )
        if not requested:
            return configured
        lookup = {
            f"{family}/{target}".lower(): (family, target)
            for family, target in configured
        }
        selected = []
        for value in requested:
            normalized = value.replace("\\", "/").strip().lower()
            try:
                selected.append(lookup[normalized])
            except KeyError as exc:
                raise KeyError(f"Unknown or unselected target: {value}") from exc
        return tuple(dict.fromkeys(selected))

    def _tasks(
        self,
        profile_name: str,
        protocols: Sequence[str],
        variants: Sequence[str],
        families: Sequence[str],
        targets: Sequence[Tuple[str, str]],
        seeds: Sequence[int],
        folds: Sequence[int],
    ) -> Tuple[ValidationTask, ...]:
        profile = self.config.profile(profile_name)
        tasks = []
        for protocol in protocols:
            if protocol == "source_cv":
                if not profile.source_cv.enabled:
                    raise ValueError(f"source_cv is disabled in profile {profile_name}")
                for variant in variants:
                    for family in families:
                        for seed in seeds:
                            for fold in folds:
                                tasks.append(
                                    ValidationTask(
                                        protocol=protocol,
                                        variant=variant,
                                        family=family,
                                        seed=seed,
                                        fold=fold,
                                    )
                                )
            elif protocol == "cpdp":
                if not profile.cpdp.enabled:
                    raise ValueError(f"cpdp is disabled in profile {profile_name}")
                for variant in variants:
                    for family, target in targets:
                        for seed in seeds:
                            tasks.append(
                                ValidationTask(
                                    protocol=protocol,
                                    variant=variant,
                                    family=family,
                                    target=target,
                                    seed=seed,
                                )
                            )
            else:
                raise ValueError(f"Unknown validation protocol: {protocol}")
        return tuple(tasks)

    def run(
        self,
        profile_name: str,
        *,
        protocols: Sequence[str] = ("source_cv", "cpdp"),
        variants: Optional[Sequence[str]] = None,
        families: Optional[Sequence[str]] = None,
        targets: Optional[Sequence[str]] = None,
        seeds: Optional[Sequence[int]] = None,
        folds: Optional[Sequence[int]] = None,
        max_tasks: Optional[int] = None,
        force: bool = False,
        fail_fast: bool = False,
    ) -> ValidationRunReport:
        profile = self.config.profile(profile_name)
        selected_variants = self._selected_variants(profile_name, variants)
        selected_seeds = self._selected_seeds(profile_name, seeds)
        selected_folds = self._selected_folds(profile_name, folds)
        selected_families = self._selected_families(families)
        selected_targets = self._selected_targets(selected_families, targets)
        tasks = self._tasks(
            profile_name,
            tuple(dict.fromkeys(protocols)),
            selected_variants,
            selected_families,
            selected_targets,
            selected_seeds,
            selected_folds,
        )
        base_config = self._base_oversampling_config(profile_name)
        fingerprint = experiment_fingerprint(
            self.config, profile_name, base_config
        )
        store = ValidationStore(self.config.output_dir, fingerprint, profile_name)
        pending = []
        skipped = 0
        for task in tasks:
            path = store.run_path(
                task.protocol,
                task.variant,
                task.family,
                task.seed,
                fold=task.fold,
                target=task.target,
            )
            if not force and store.is_complete(path):
                skipped += 1
            else:
                pending.append(task)
        if max_tasks is not None:
            if max_tasks < 1:
                raise ValueError("max_tasks must be positive")
            pending = pending[:max_tasks]

        if not pending:
            if skipped:
                try:
                    store.summarize()
                except RuntimeError:
                    pass
            return ValidationRunReport(
                fingerprint=fingerprint,
                selected=len(tasks),
                attempted=0,
                succeeded=0,
                failed=0,
                skipped=skipped,
                output_root=store.root,
            )

        projects = load_all_projects(self.config.experiment.data)
        family_projects = {
            family.name: tuple(
                projects[f"{family.name}/{name}"] for name in family.projects
            )
            for family in self.config.experiment.data.families
        }
        succeeded = 0
        failed = 0
        for index, task in enumerate(pending, start=1):
            path = store.run_path(
                task.protocol,
                task.variant,
                task.family,
                task.seed,
                fold=task.fold,
                target=task.target,
            )
            variant = self.config.variant(task.variant)
            strategy = create_strategy(variant.kind, base_config, variant.overrides)
            common = {
                "fingerprint": fingerprint,
                "profile": profile_name,
                "protocol": task.protocol,
                "variant": task.variant,
                "variant_kind": variant.kind,
                "family": task.family,
                "target": task.target,
                "seed": task.seed,
                "fold": task.fold,
                "started_at": _utc_now(),
            }
            print(
                "VALIDATION "
                f"task={index}/{len(pending)} protocol={task.protocol} "
                f"variant={task.variant} family={task.family} "
                f"target={task.target or '-'} seed={task.seed} "
                f"fold={task.fold if task.fold is not None else '-'}",
                flush=True,
            )
            try:
                seed_everything(task.seed)
                if task.protocol == "source_cv":
                    result = run_source_cv_fold(
                        family_projects[task.family],
                        strategy,
                        profile,
                        task.seed,
                        int(task.fold),
                    )
                else:
                    target_project = projects[f"{task.family}/{task.target}"]
                    result = run_cpdp_target(
                        tuple(
                            sources_for_target(
                                projects, task.family, str(task.target)
                            )
                        ),
                        target_project,
                        strategy,
                        profile,
                        task.seed,
                    )
                store.save(
                    path,
                    {
                        **common,
                        "status": "success",
                        "completed_at": _utc_now(),
                        "utility": result.utility,
                        "quality": result.quality,
                    },
                )
                succeeded += 1
            except Exception as exc:
                failed += 1
                store.save(
                    path,
                    {
                        **common,
                        "status": "failure",
                        "completed_at": _utc_now(),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
                print(
                    f"VALIDATION_FAILURE path={path} error={type(exc).__name__}: {exc}",
                    flush=True,
                )
                if fail_fast:
                    raise
        if succeeded or skipped:
            try:
                store.summarize()
            except RuntimeError:
                pass
        return ValidationRunReport(
            fingerprint=fingerprint,
            selected=len(tasks),
            attempted=len(pending),
            succeeded=succeeded,
            failed=failed,
            skipped=skipped,
            output_root=store.root,
        )

    def summarize(self, profile_name: str) -> Mapping[str, Path]:
        fingerprint = self.fingerprint(profile_name)
        return ValidationStore(
            self.config.output_dir, fingerprint, profile_name
        ).summarize()
