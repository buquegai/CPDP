from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from cpdp.oversampling.config import OversamplingConfig
from cpdp.oversampling.diffusion import ProjectConditionedTabularDDPM
from cpdp.oversampling.formula_miner import (
    FrozenFormulaGuidance,
    SparseFormulaMiner,
)
from cpdp.oversampling.preprocessing import RobustAffineScaler
from cpdp.oversampling.semantic_roles import SemanticRoleResolver
from cpdp.oversampling.symbolic_library import FittedSymbolicLibrary
from cpdp.oversampling.types import (
    AugmentationResult,
    ProjectGenerationDiagnostics,
    freeze_array,
)
from cpdp.types import LabeledProject, ProjectFeatures


class IPGDOOversampler:
    """Implicit Physics-Guided Diffusion Oversampling for source-only CPDP."""

    def __init__(self, config: OversamplingConfig) -> None:
        self.config = config

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "IPGDOOversampler":
        return cls(OversamplingConfig.from_mapping(values))

    def fit_resample(
        self, sources: Sequence[LabeledProject], seed: int
    ) -> AugmentationResult:
        projects = tuple(sources)
        self._validate_sources(projects)
        requested = tuple(self._requested_count(project) for project in projects)

        if sum(requested) == 0:
            return AugmentationResult(
                projects=projects,
                metadata={
                    "enabled": False,
                    "name": "IPGDO",
                    "display_name": "Implicit Physics-Guided Diffusion Oversampling",
                    "reason": "all_source_projects_already_meet_target_ratio",
                    "target_data_used": False,
                    "configuration": asdict(self.config),
                    "total_synthetic_selected": 0,
                },
            )

        all_values = np.vstack([project.features.values for project in projects])
        all_labels = np.concatenate([project.labels for project in projects])
        formula_guidance, formula_metadata = self._fit_formula_guidance(
            projects, all_values, all_labels
        )

        defect_batches = tuple(
            project.features.values[project.labels == 1] for project in projects
        )
        project_scalers = tuple(
            RobustAffineScaler.fit(defects) for defects in defect_batches
        )
        normalized_defects = np.vstack(
            [
                np.clip(
                    scaler.transform(defects),
                    -self.config.diffusion.normalized_clip,
                    self.config.diffusion.normalized_clip,
                )
                for defects, scaler in zip(defect_batches, project_scalers)
            ]
        ).astype(np.float32)
        raw_defects = np.vstack(defect_batches).astype(np.float32)
        defect_project_ids = np.concatenate(
            [
                np.full(len(defects), index, dtype=np.int64)
                for index, defects in enumerate(defect_batches)
            ]
        )
        scaler_centers = np.vstack([scaler.center for scaler in project_scalers])
        scaler_scales = np.vstack([scaler.scale for scaler in project_scalers])

        diffusion = ProjectConditionedTabularDDPM(
            input_dim=normalized_defects.shape[1],
            project_count=len(projects),
            config=self.config.diffusion,
            device=self.config.device,
        )
        diffusion_metadata = diffusion.fit(
            normalized_defects,
            defect_project_ids,
            seed,
            raw_defects=raw_defects,
            scaler_centers=scaler_centers,
            scaler_scales=scaler_scales,
            formula_guidance=formula_guidance,
        )

        augmented: List[LabeledProject] = []
        diagnostics: List[ProjectGenerationDiagnostics] = []
        for project_index, (project, needed) in enumerate(zip(projects, requested)):
            clean_count = int(np.sum(project.labels == 0))
            defect_count = int(np.sum(project.labels == 1))
            if needed == 0:
                augmented.append(project)
                diagnostics.append(
                    ProjectGenerationDiagnostics(
                        project=project.key,
                        clean_samples=clean_count,
                        real_defects=defect_count,
                        requested_synthetic=0,
                        generation_rounds=0,
                        generated_candidates=0,
                        selected_synthetic=0,
                        formula_guidance=formula_metadata,
                    )
                )
                continue

            generated_normalized = diffusion.sample_defects(needed, project_index)
            generated_raw = project_scalers[project_index].inverse_transform(
                generated_normalized
            )
            if not np.isfinite(generated_raw).all():
                raise RuntimeError(
                    f"IPGDO generated non-finite values for {project.key}"
                )

            augmented.append(self._append_synthetic(project, generated_raw))
            diagnostics.append(
                ProjectGenerationDiagnostics(
                    project=project.key,
                    clean_samples=clean_count,
                    real_defects=defect_count,
                    requested_synthetic=needed,
                    generation_rounds=1,
                    generated_candidates=needed,
                    selected_synthetic=int(len(generated_raw)),
                    formula_guidance=formula_metadata,
                )
            )
            print(
                "PROGRESS ipgdo "
                f"project={project.key} requested={needed} "
                f"generated={len(generated_raw)} selected={len(generated_raw)}",
                flush=True,
            )

        return AugmentationResult(
            projects=tuple(augmented),
            metadata={
                "enabled": True,
                "name": "IPGDO",
                "display_name": "Implicit Physics-Guided Diffusion Oversampling",
                "target_data_used": False,
                "training_data": "real_source_defects_only",
                "architecture": [
                    "project-conditioned tabular DDPM",
                    "adaptive semantic-role discovery",
                    "sparse PSE symbolic formula mining",
                    "frozen formula-guided diffusion training",
                ],
                "selection": {
                    "enabled": False,
                    "method": "none",
                    "reason": "no_candidate_selection",
                },
                "formula_guidance": formula_metadata,
                "configuration": asdict(self.config),
                "project_scalers": {
                    project.key: scaler.as_dict()
                    for project, scaler in zip(projects, project_scalers)
                },
                "diffusion": diffusion_metadata,
                "projects": [asdict(item) for item in diagnostics],
                "total_synthetic_selected": int(
                    sum(item.selected_synthetic for item in diagnostics)
                ),
            },
        )

    def _fit_formula_guidance(
        self,
        projects: Tuple[LabeledProject, ...],
        all_values: np.ndarray,
        all_labels: np.ndarray,
    ) -> Tuple[Any, Dict[str, Any]]:
        try:
            role_mapping = SemanticRoleResolver(self.config.role).fit(
                all_values,
                all_labels,
                projects[0].features.feature_names,
            )
            library = FittedSymbolicLibrary.fit(all_values, role_mapping)
            formula_bank = SparseFormulaMiner(self.config.formula).fit(
                projects, library
            )
            guidance = FrozenFormulaGuidance(
                library,
                formula_bank,
                use_reliability_weighting=self.config.formula.use_reliability_weighting,
            )
            metadata = {
                "enabled": bool(formula_bank.enabled),
                "source_only": True,
                "role_mapping": role_mapping.as_dict(),
                "formula_bank": formula_bank.as_dict(),
            }
            return (guidance if formula_bank.enabled else None), metadata
        except (FloatingPointError, ValueError) as exc:
            metadata = {
                "enabled": False,
                "source_only": True,
                "reason": f"formula_discovery_unavailable: {exc}",
            }
            return None, metadata

    def _requested_count(self, project: LabeledProject) -> int:
        clean_count = int(np.sum(project.labels == 0))
        defect_count = int(np.sum(project.labels == 1))
        return max(
            0,
            int(math.ceil(clean_count * self.config.target_minority_ratio))
            - defect_count,
        )

    @staticmethod
    def _validate_sources(projects: Tuple[LabeledProject, ...]) -> None:
        if not projects:
            raise ValueError(
                "Diffusion oversampling requires at least one source project"
            )
        family = projects[0].features.family
        schema = projects[0].features.feature_names
        keys = set()
        for project in projects:
            if project.key in keys:
                raise ValueError(f"Duplicate source project: {project.key}")
            keys.add(project.key)
            if project.features.family != family:
                raise ValueError("All source projects must belong to one dataset family")
            if project.features.feature_names != schema:
                raise ValueError("All source projects must use the same feature schema")
            if set(np.unique(project.labels).tolist()) != {0, 1}:
                raise ValueError(f"Source project must contain both classes: {project.key}")
            if int(np.sum(project.labels == 1)) < 1:
                raise ValueError(
                    f"Source project needs at least one defect: {project.key}"
                )

    @staticmethod
    def _append_synthetic(
        project: LabeledProject, synthetic_values: np.ndarray
    ) -> LabeledProject:
        if len(synthetic_values) == 0:
            return project
        values = np.vstack((project.features.values, synthetic_values))
        labels = np.concatenate(
            (project.labels, np.ones(len(synthetic_values), dtype=np.int64))
        )
        effort_index = project.features.feature_names.index(
            project.features.effort_name
        )
        effort = np.concatenate(
            (project.features.effort, synthetic_values[:, effort_index])
        )
        synthetic_row_ids = -np.arange(1, len(synthetic_values) + 1, dtype=np.int64)
        row_ids = np.concatenate((project.features.row_ids, synthetic_row_ids))
        cleaning = dict(project.features.cleaning)
        cleaning["synthetic_rows_added"] = int(len(synthetic_values))
        features = ProjectFeatures(
            family=project.features.family,
            name=project.features.name,
            values=freeze_array(values, np.float64),
            feature_names=project.features.feature_names,
            effort=freeze_array(effort, np.float64),
            effort_name=project.features.effort_name,
            row_ids=freeze_array(row_ids, np.int64),
            cleaning=cleaning,
        )
        return LabeledProject(features=features, labels=freeze_array(labels, np.int64))
