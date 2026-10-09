from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.preprocessing import RobustScaler

from cpdp.predictors.base import BasePredictor
from cpdp.predictors.jepa_losses import (
    latent_prediction_loss,
    sample_feature_mask,
    variance_covariance_losses,
    weighted_coral_loss,
)
from cpdp.predictors.kan import (
    KANEncoderConfig,
    build_kan_jepa_model,
    update_ema_target,
)
from cpdp.predictors.tabr import TabRClassifierConfig
from cpdp.predictors.threshold import select_metric_threshold
from cpdp.types import LabeledProject, PredictionResult, UnlabeledProject


_BATCH_LOSS_LOG_ENV = "CPDP_PREDICTOR_BATCH_LOSS_LOG"
_batch_loss_log_path: Optional[Path] = None
_batch_loss_log_handle = None


def _record_batch_losses(record: Mapping[str, Any]) -> None:
    global _batch_loss_log_handle, _batch_loss_log_path
    raw_path = os.environ.get(_BATCH_LOSS_LOG_ENV, "").strip()
    if not raw_path:
        return
    path = Path(raw_path).expanduser().resolve()
    if path != _batch_loss_log_path or not path.exists():
        if _batch_loss_log_handle is not None:
            _batch_loss_log_handle.close()
        path.parent.mkdir(parents=True, exist_ok=True)
        _batch_loss_log_handle = path.open(
            "w", encoding="utf-8", newline="\n", buffering=1024 * 1024
        )
        _batch_loss_log_path = path
    _batch_loss_log_handle.write(
        json.dumps(dict(record), ensure_ascii=False, separators=(",", ":")) + "\n"
    )


def _flush_batch_loss_log() -> None:
    if _batch_loss_log_handle is not None:
        _batch_loss_log_handle.flush()


def _batch_loss_log_metadata() -> Dict[str, Any]:
    raw_path = os.environ.get(_BATCH_LOSS_LOG_ENV, "").strip()
    return {
        "enabled": bool(raw_path),
        "format": "jsonl",
        "granularity": "optimizer-batch",
        "path": str(Path(raw_path).expanduser().resolve()) if raw_path else None,
    }


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _probability(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


@dataclass(frozen=True)
class JEPAPredictorConfig:
    hidden_dims: Tuple[int, ...] = (128,)
    mask_embedding_dim: int = 32
    dropout: float = 0.1

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "JEPAPredictorConfig":
        config = cls(
            hidden_dims=tuple(int(value) for value in values.get("hidden_dims", (128,))),
            mask_embedding_dim=int(values.get("mask_embedding_dim", 32)),
            dropout=float(values.get("dropout", 0.1)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.hidden_dims or min(self.hidden_dims) < 1:
            raise ValueError("jepa_predictor.hidden_dims must contain positive dimensions")
        if self.mask_embedding_dim < 1:
            raise ValueError("jepa_predictor.mask_embedding_dim must be positive")
        _probability(self.dropout, "jepa_predictor.dropout")


@dataclass(frozen=True)
class ViewConfig:
    minimum_mask_ratio: float = 0.2
    maximum_mask_ratio: float = 0.5
    target_noise_std: float = 0.01
    consistency_noise_std: float = 0.02

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ViewConfig":
        config = cls(
            minimum_mask_ratio=float(values.get("minimum_mask_ratio", 0.2)),
            maximum_mask_ratio=float(values.get("maximum_mask_ratio", 0.5)),
            target_noise_std=float(values.get("target_noise_std", 0.01)),
            consistency_noise_std=float(values.get("consistency_noise_std", 0.02)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        _probability(self.minimum_mask_ratio, "views.minimum_mask_ratio")
        _probability(self.maximum_mask_ratio, "views.maximum_mask_ratio")
        if self.minimum_mask_ratio > self.maximum_mask_ratio:
            raise ValueError("view mask ratios must be increasing")
        if self.maximum_mask_ratio >= 1.0:
            raise ValueError("views.maximum_mask_ratio must be less than 1")
        if self.target_noise_std < 0 or self.consistency_noise_std < 0:
            raise ValueError("view noise standard deviations cannot be negative")


@dataclass(frozen=True)
class LossWeights:
    jepa: float = 1.0
    classification: float = 1.0
    alignment: float = 0.1
    consistency: float = 0.1
    variance: float = 0.05
    covariance: float = 0.01

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "LossWeights":
        config = cls(
            jepa=float(values.get("jepa", 1.0)),
            classification=float(values.get("classification", 1.0)),
            alignment=float(values.get("alignment", 0.1)),
            consistency=float(values.get("consistency", 0.1)),
            variance=float(values.get("variance", 0.05)),
            covariance=float(values.get("covariance", 0.01)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if value < 0:
                raise ValueError(f"losses.{name} cannot be negative")
        if self.jepa == 0 or self.classification == 0:
            raise ValueError("JEPA and classification losses must remain enabled")


@dataclass(frozen=True)
class OptimizationConfig:
    pretrain_epochs: int = 80
    joint_epochs: int = 150
    batch_size: int = 256
    learning_rate: float = 5e-4
    learning_rate_schedule: str = "constant"
    learning_rate_warmup_start: float = 2e-5
    learning_rate_peak: float = 1.5e-4
    learning_rate_min: float = 1e-5
    learning_rate_warmup_fraction: float = 0.1
    learning_rate_decay_epochs: int = 50
    weight_decay: float = 1e-5
    ema_momentum: float = 0.996
    early_stopping_patience: int = 25
    gradient_clip_norm: float = 1.0
    source_weight_temperature: float = 1.0
    source_weight_update_interval: int = 5
    use_amp: bool = True
    refit_all_sources: bool = True
    log_every_epochs: int = 10

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "OptimizationConfig":
        base_learning_rate = float(values.get("learning_rate", 5e-4))
        config = cls(
            pretrain_epochs=int(values.get("pretrain_epochs", 80)),
            joint_epochs=int(values.get("joint_epochs", 150)),
            batch_size=int(values.get("batch_size", 256)),
            learning_rate=base_learning_rate,
            learning_rate_schedule=str(
                values.get("learning_rate_schedule", "constant")
            ).strip().lower(),
            learning_rate_warmup_start=float(
                values.get("learning_rate_warmup_start", min(base_learning_rate, 2e-5))
            ),
            learning_rate_peak=float(
                values.get("learning_rate_peak", base_learning_rate)
            ),
            learning_rate_min=float(
                values.get("learning_rate_min", base_learning_rate)
            ),
            learning_rate_warmup_fraction=float(
                values.get("learning_rate_warmup_fraction", 0.1)
            ),
            learning_rate_decay_epochs=int(
                values.get("learning_rate_decay_epochs", 50)
            ),
            weight_decay=float(values.get("weight_decay", 1e-5)),
            ema_momentum=float(values.get("ema_momentum", 0.996)),
            early_stopping_patience=int(values.get("early_stopping_patience", 25)),
            gradient_clip_norm=float(values.get("gradient_clip_norm", 1.0)),
            source_weight_temperature=float(values.get("source_weight_temperature", 1.0)),
            source_weight_update_interval=int(values.get("source_weight_update_interval", 5)),
            use_amp=bool(values.get("use_amp", True)),
            refit_all_sources=bool(values.get("refit_all_sources", True)),
            log_every_epochs=int(values.get("log_every_epochs", 10)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.pretrain_epochs < 1 or self.joint_epochs < 1:
            raise ValueError("training epochs must be positive")
        if self.batch_size < 2:
            raise ValueError("optimization.batch_size must be at least 2")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("optimizer values are invalid")
        if self.learning_rate_schedule not in {"constant", "warmup_cosine"}:
            raise ValueError(
                "optimization.learning_rate_schedule must be constant or warmup_cosine"
            )
        for name, value in (
            ("learning_rate_warmup_start", self.learning_rate_warmup_start),
            ("learning_rate_peak", self.learning_rate_peak),
            ("learning_rate_min", self.learning_rate_min),
        ):
            if value <= 0:
                raise ValueError(f"optimization.{name} must be positive")
        if self.learning_rate_warmup_start > self.learning_rate_peak:
            raise ValueError(
                "optimization.learning_rate_warmup_start cannot exceed learning_rate_peak"
            )
        if self.learning_rate_min > self.learning_rate_peak:
            raise ValueError(
                "optimization.learning_rate_min cannot exceed learning_rate_peak"
            )
        if not 0.0 <= self.learning_rate_warmup_fraction < 1.0:
            raise ValueError(
                "optimization.learning_rate_warmup_fraction must be in [0, 1)"
            )
        if self.learning_rate_decay_epochs < 1:
            raise ValueError(
                "optimization.learning_rate_decay_epochs must be positive"
            )
        if not 0.0 < self.ema_momentum < 1.0:
            raise ValueError("optimization.ema_momentum must be between 0 and 1")
        if self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be positive")
        if self.gradient_clip_norm <= 0 or self.source_weight_temperature <= 0:
            raise ValueError("gradient clip and source temperature must be positive")
        if self.source_weight_update_interval < 1 or self.log_every_epochs < 1:
            raise ValueError("optimization intervals must be positive")


@dataclass(frozen=True)
class KANJEPAConfig:
    device: str = "auto"
    input_clip: float = 8.0
    data_flow_mode: str = "split"
    encoder: KANEncoderConfig = field(default_factory=KANEncoderConfig)
    jepa_predictor: JEPAPredictorConfig = field(default_factory=JEPAPredictorConfig)
    classifier: TabRClassifierConfig = field(default_factory=TabRClassifierConfig)
    views: ViewConfig = field(default_factory=ViewConfig)
    losses: LossWeights = field(default_factory=LossWeights)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    threshold_policy: str = "mcc"
    probability_calibration: str = "none"

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "KANJEPAConfig":
        config = cls(
            device=str(values.get("device", "auto")),
            input_clip=float(values.get("input_clip", 8.0)),
            data_flow_mode=str(values.get("data_flow_mode", "split")).strip().lower(),
            encoder=KANEncoderConfig.from_mapping(_mapping(values.get("encoder"), "encoder")),
            jepa_predictor=JEPAPredictorConfig.from_mapping(
                _mapping(values.get("jepa_predictor"), "jepa_predictor")
            ),
            classifier=TabRClassifierConfig.from_mapping(
                _mapping(values.get("classifier"), "classifier")
            ),
            views=ViewConfig.from_mapping(_mapping(values.get("views"), "views")),
            losses=LossWeights.from_mapping(_mapping(values.get("losses"), "losses")),
            optimization=OptimizationConfig.from_mapping(
                _mapping(values.get("optimization"), "optimization")
            ),
            threshold_policy=str(values.get("threshold_policy", "mcc")).strip().lower(),
            probability_calibration=str(
                values.get("probability_calibration", "none")
            ).strip().lower(),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("predictor.device must be 'auto', 'cpu', or 'cuda'")
        if self.input_clip <= 0:
            raise ValueError("predictor.input_clip must be positive")
        if self.data_flow_mode not in {"split", "shared_augmented_source"}:
            raise ValueError(
                "predictor.data_flow_mode must be 'split' or " 
                "'shared_augmented_source'"
            )
        if self.threshold_policy not in {"mcc", "f1", "gmean", "fixed_0.5"}:
            raise ValueError(
                "predictor.threshold_policy must be mcc, f1, gmean, or fixed_0.5"
            )
        if self.probability_calibration not in {"none", "platt"}:
            raise ValueError(
                "predictor.probability_calibration must be none or platt"
            )


@dataclass(frozen=True)
class _ProjectPartition:
    key: str
    real_values: np.ndarray
    real_labels: np.ndarray
    augmented_values: np.ndarray
    augmented_labels: np.ndarray


@dataclass(frozen=True)
class _TrainingArrays:
    project_names: Tuple[str, ...]
    real_values_by_project: Tuple[np.ndarray, ...]
    real_labels_by_project: Tuple[np.ndarray, ...]
    real_values: np.ndarray
    real_project_ids: np.ndarray
    augmented_values: np.ndarray
    augmented_labels: np.ndarray
    augmented_project_ids: np.ndarray
    target_values: np.ndarray
    scaler: RobustScaler


@dataclass
class _TrainingOutcome:
    model: Any
    arrays: _TrainingArrays
    completed_pretrain_epochs: int
    completed_joint_epochs: int
    best_joint_epoch: int
    best_validation_loss: Optional[float]
    final_losses: Dict[str, float]
    source_weights: np.ndarray


def _resolve_device(torch, requested: str, model_name: str):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested for {model_name}, but it is unavailable")
    return torch.device(requested)


def _copy_state_to_cpu(model) -> Dict[str, Any]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.state_dict().items()
    }


def _cycle(loader):
    while True:
        yield from loader


class JEPAPredictorBase(BasePredictor):
    """Shared transductive JEPA training and CPDP evaluation protocol."""

    model_name = "JEPA"
    metadata_name = "jepa"
    progress_name = "jepa"

    def __init__(self, config: Any) -> None:
        self.config = config

    def fit_predict(
        self,
        sources: Sequence[LabeledProject],
        target: UnlabeledProject,
        seed: int,
    ) -> PredictionResult:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(f"The {self.model_name} predictor requires PyTorch") from exc
        partitions = self._partition_sources(sources, target)
        if len(partitions) < 2:
            raise ValueError(f"{self.model_name} requires at least two source projects")
        device = _resolve_device(torch, self.config.device, self.model_name)
        validation_index = int(
            np.random.default_rng(seed).integers(0, len(partitions))
        )
        validation_project = partitions[validation_index]
        selection_projects = tuple(
            project for index, project in enumerate(partitions) if index != validation_index
        )
        selection_seed = int(seed + 104729)
        selection = self._train(
            torch,
            selection_projects,
            target,
            device,
            selection_seed,
            self.config.optimization.joint_epochs,
            validation_project,
            "selection",
        )
        validation_values = self._transform(
            selection.arrays.scaler, validation_project.real_values
        )
        validation_scores = self._scores(
            torch, selection.model, validation_values, selection.arrays, device
        )
        calibrated_validation_scores = validation_scores
        calibrator = None
        if getattr(self.config, "probability_calibration", "none") == "platt":
            from sklearn.linear_model import LogisticRegression

            calibrator = LogisticRegression(
                C=1.0, solver="lbfgs", max_iter=1000, random_state=selection_seed
            )
            calibrator.fit(
                validation_scores.reshape(-1, 1), validation_project.real_labels
            )
            calibrated_validation_scores = calibrator.predict_proba(
                validation_scores.reshape(-1, 1)
            )[:, 1]
        threshold_policy = getattr(self.config, "threshold_policy", "mcc")
        threshold = select_metric_threshold(
            validation_project.real_labels,
            calibrated_validation_scores,
            threshold_policy,
        )

        if self.config.optimization.refit_all_sources:
            final_seed = int(seed + 209759)
            final = self._train(
                torch,
                partitions,
                target,
                device,
                final_seed,
                selection.best_joint_epoch,
                None,
                "final",
            )
        else:
            final = selection
            final_seed = selection_seed
        scores = self._scores(
            torch, final.model, final.arrays.target_values, final.arrays, device
        )
        if calibrator is not None:
            scores = calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]
        synthetic_rows = sum(
            len(project.augmented_values) - len(project.real_values)
            for project in partitions
        )
        parameter_count = sum(
            parameter.numel()
            for parameter in final.model.parameters()
            if parameter.requires_grad
        )
        return PredictionResult(
            scores=scores,
            threshold=threshold,
            metadata={
                "predictor": self.metadata_name,
                "device": str(device),
                "amp_enabled": bool(
                    self.config.optimization.use_amp and device.type == "cuda"
                ),
                "configuration": asdict(self.config),
                "batch_loss_logging": _batch_loss_log_metadata(),
                "architecture": {
                    **self._architecture_metadata(),
                    "classifier": self._classifier_name(),
                    "trainable_parameters": int(parameter_count),
                },
                "data_contract": {
                    "jepa": (
                        "augmented-source only"
                        if self.config.data_flow_mode == "shared_augmented_source"
                        else "real-source plus unlabeled-target only"
                    ),
                    "classification": "real-source plus synthetic-source",
                    "classification_context": self._classification_context_name(),
                    "alignment": "real-source to unlabeled-target only",
                    "data_flow_mode": self.config.data_flow_mode,
                    "synthetic_rows": int(synthetic_rows),
                    "target_features_used": True,
                    "target_labels_used": False,
                    "pseudo_labels_used": False,
                },
                "normalization": {
                    "name": "source-real-only RobustScaler",
                    "target_statistics_used": False,
                    "synthetic_statistics_used": False,
                    "input_clip": self.config.input_clip,
                },
                "model_selection": {
                    "policy": "seeded leave-one-source-project-out",
                    "validation_project": validation_project.key,
                    "validation_rows": int(len(validation_project.real_labels)),
                    "threshold_policy": f"source-validation-{threshold_policy}",
                    "probability_calibration": getattr(self.config, "probability_calibration", "none"),
                    "threshold": threshold,
                    "selection_seed": selection_seed,
                    "best_joint_epoch": selection.best_joint_epoch,
                    "best_validation_loss": selection.best_validation_loss,
                    "validation_score_mean": float(np.mean(validation_scores)),
                    "validation_score_std": float(np.std(validation_scores)),
                    "calibrated_validation_score_mean": float(
                        np.mean(calibrated_validation_scores)
                    ),
                    "calibrated_validation_score_std": float(
                        np.std(calibrated_validation_scores)
                    ),
                },
                "final_training": {
                    "refit_all_sources": self.config.optimization.refit_all_sources,
                    "seed": final_seed,
                    "pretrain_epochs": final.completed_pretrain_epochs,
                    "joint_epochs": final.completed_joint_epochs,
                    "real_source_rows": int(len(final.arrays.real_values)),
                    "augmented_source_rows": int(len(final.arrays.augmented_values)),
                    "target_rows": int(len(final.arrays.target_values)),
                    "source_weights": {
                        name: float(weight)
                        for name, weight in zip(
                            final.arrays.project_names, final.source_weights
                        )
                    },
                    "final_losses": final.final_losses,
                },
            },
        )

    def _partition_sources(
        self,
        sources: Sequence[LabeledProject],
        target: UnlabeledProject,
    ) -> Tuple[_ProjectPartition, ...]:
        if not sources:
            raise ValueError(f"{self.model_name} requires source projects")
        schema = target.features.feature_names
        family = target.features.family
        partitions = []
        keys = set()
        for source in sources:
            if source.key in keys:
                raise ValueError(f"Duplicate source project: {source.key}")
            keys.add(source.key)
            if source.key == target.key:
                raise ValueError("Held-out target cannot appear among source projects")
            if source.features.family != family:
                raise ValueError("Sources and target must belong to one dataset family")
            if source.features.feature_names != schema:
                raise ValueError("Sources and target must share one feature schema")
            real_mask = np.asarray(source.features.row_ids) >= 0
            if not real_mask.any():
                raise ValueError(f"Source project has no real rows: {source.key}")
            real_labels = np.asarray(source.labels[real_mask], dtype=np.int64)
            if set(np.unique(real_labels).tolist()) != {0, 1}:
                raise ValueError(f"Real source project must contain both classes: {source.key}")
            partitions.append(
                _ProjectPartition(
                    key=source.key,
                    real_values=np.asarray(
                        source.features.values[real_mask], dtype=np.float64
                    ),
                    real_labels=real_labels,
                    augmented_values=np.asarray(source.features.values, dtype=np.float64),
                    augmented_labels=np.asarray(source.labels, dtype=np.int64),
                )
            )
        return tuple(partitions)

    def _prepare_arrays(
        self,
        projects: Sequence[_ProjectPartition],
        target: UnlabeledProject,
    ) -> _TrainingArrays:
        raw_real = np.vstack([project.real_values for project in projects])
        scaler = RobustScaler(quantile_range=(25.0, 75.0)).fit(raw_real)
        real_by_project = tuple(
            self._transform(scaler, project.real_values) for project in projects
        )
        labels_by_project = tuple(project.real_labels for project in projects)
        augmented_by_project = tuple(
            self._transform(scaler, project.augmented_values) for project in projects
        )
        real_project_ids = np.concatenate(
            [np.full(len(values), index, dtype=np.int64) for index, values in enumerate(real_by_project)]
        )
        augmented_project_ids = np.concatenate(
            [
                np.full(len(values), index, dtype=np.int64)
                for index, values in enumerate(augmented_by_project)
            ]
        )
        return _TrainingArrays(
            project_names=tuple(project.key for project in projects),
            real_values_by_project=real_by_project,
            real_labels_by_project=labels_by_project,
            real_values=np.vstack(real_by_project).astype(np.float32),
            real_project_ids=real_project_ids,
            augmented_values=np.vstack(augmented_by_project).astype(np.float32),
            augmented_labels=np.concatenate(
                [project.augmented_labels for project in projects]
            ).astype(np.float32),
            augmented_project_ids=augmented_project_ids,
            target_values=self._transform(
                scaler, np.asarray(target.features.values, dtype=np.float64)
            ),
            scaler=scaler,
        )

    def _transform(self, scaler: RobustScaler, values: np.ndarray) -> np.ndarray:
        transformed = scaler.transform(values)
        transformed = np.nan_to_num(
            transformed,
            nan=0.0,
            posinf=self.config.input_clip,
            neginf=-self.config.input_clip,
        )
        return np.clip(
            transformed, -self.config.input_clip, self.config.input_clip
        ).astype(np.float32)

    def _train(
        self,
        torch,
        projects: Sequence[_ProjectPartition],
        target: UnlabeledProject,
        device,
        seed: int,
        joint_epochs: int,
        validation_project: Optional[_ProjectPartition],
        stage_name: str,
    ) -> _TrainingOutcome:
        self._seed_torch(torch, seed)
        arrays = self._prepare_arrays(projects, target)
        model = self._build_model(torch, arrays.real_values.shape[1]).to(device)
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable,
            lr=self.config.optimization.learning_rate,
            weight_decay=self.config.optimization.weight_decay,
        )
        amp_enabled = bool(
            self.config.optimization.use_amp and device.type == "cuda"
        )
        gradient_scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        loaders = self._loaders(torch, arrays, seed, device.type == "cuda")
        pretrain_schedule_epochs = min(
            self.config.optimization.pretrain_epochs,
            self.config.optimization.learning_rate_decay_epochs,
        )
        pretrain_steps = max(pretrain_schedule_epochs * len(loaders["ssl"]), 1)
        joint_schedule_epochs = min(
            joint_epochs,
            self.config.optimization.learning_rate_decay_epochs,
        )
        joint_steps = max(
            joint_schedule_epochs * max(len(loader) for loader in loaders.values()),
            1,
        )
        classification_context = self._classification_context(
            torch, arrays, device
        )
        final_losses: Dict[str, float] = {}

        for epoch in range(self.config.optimization.pretrain_epochs):
            final_losses = self._pretrain_epoch(
                torch,
                model,
                optimizer,
                gradient_scaler,
                loaders["ssl"],
                device,
                amp_enabled,
                stage_name,
                epoch + 1,
                self.config.optimization.pretrain_epochs,
                pretrain_steps,
            )
            self._log_epoch(
                stage_name, "pretrain", epoch + 1, self.config.optimization.pretrain_epochs, final_losses
            )

        validation_values = None
        validation_labels = None
        if validation_project is not None:
            validation_values = self._transform(arrays.scaler, validation_project.real_values)
            validation_labels = validation_project.real_labels.astype(np.float32)
        best_state = None
        best_loss = float("inf")
        best_epoch = joint_epochs
        stale_epochs = 0
        source_weights = np.full(
            len(arrays.project_names), 1.0 / len(arrays.project_names), dtype=np.float64
        )
        completed_joint_epochs = 0

        for epoch in range(joint_epochs):
            if epoch % self.config.optimization.source_weight_update_interval == 0:
                source_weights = self._source_project_weights(
                    torch, model, arrays, device
                )
            final_losses = self._joint_epoch(
                torch,
                model,
                optimizer,
                gradient_scaler,
                loaders,
                classification_context,
                source_weights,
                device,
                amp_enabled,
                stage_name,
                epoch + 1,
                joint_epochs,
                joint_steps,
            )
            completed_joint_epochs = epoch + 1
            if validation_values is not None and validation_labels is not None:
                validation_loss = self._validation_loss(
                    torch,
                    model,
                    validation_values,
                    validation_labels,
                    classification_context,
                    device,
                )
                final_losses["validation"] = validation_loss
                if validation_loss < best_loss - 1e-6:
                    best_loss = validation_loss
                    best_state = _copy_state_to_cpu(model)
                    best_epoch = epoch + 1
                    stale_epochs = 0
                else:
                    stale_epochs += 1
            self._log_epoch(stage_name, "joint", epoch + 1, joint_epochs, final_losses)
            if (
                validation_values is not None
                and stale_epochs >= self.config.optimization.early_stopping_patience
            ):
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        source_weights = self._source_project_weights(torch, model, arrays, device)
        return _TrainingOutcome(
            model=model,
            arrays=arrays,
            completed_pretrain_epochs=self.config.optimization.pretrain_epochs,
            completed_joint_epochs=completed_joint_epochs,
            best_joint_epoch=best_epoch,
            best_validation_loss=None if best_state is None else best_loss,
            final_losses=final_losses,
            source_weights=source_weights,
        )

    def _loaders(self, torch, arrays: _TrainingArrays, seed: int, pin_memory: bool):
        batch_size = self.config.optimization.batch_size
        augmented_weights = self._balanced_group_weights(arrays.augmented_project_ids)
        real_weights = self._balanced_group_weights(arrays.real_project_ids)
        target_group = len(arrays.project_names)
        if self.config.data_flow_mode == "shared_augmented_source":
            ssl_values = arrays.augmented_values
            ssl_groups = arrays.augmented_project_ids
        else:
            ssl_values = np.vstack((arrays.real_values, arrays.target_values)).astype(
                np.float32
            )
            ssl_groups = np.concatenate(
                (
                    arrays.real_project_ids,
                    np.full(len(arrays.target_values), target_group, dtype=np.int64),
                )
            )
        ssl_weights = self._balanced_group_weights(ssl_groups)
        return {
            "classification": self._loader(
                torch,
                (
                    arrays.augmented_values,
                    arrays.augmented_labels,
                    arrays.augmented_project_ids,
                    np.arange(len(arrays.augmented_values), dtype=np.int64),
                ),
                batch_size,
                seed + 1,
                augmented_weights,
                pin_memory,
            ),
            "real": self._loader(
                torch,
                (arrays.real_values, arrays.real_project_ids),
                batch_size,
                seed + 2,
                real_weights,
                pin_memory,
            ),
            "target": self._loader(
                torch,
                (arrays.target_values,),
                batch_size,
                seed + 3,
                None,
                pin_memory,
            ),
            "ssl": self._loader(
                torch,
                (ssl_values,),
                batch_size,
                seed + 4,
                ssl_weights,
                pin_memory,
            ),
        }

    @staticmethod
    def _balanced_group_weights(group_ids: np.ndarray) -> np.ndarray:
        counts = np.bincount(group_ids)
        return np.asarray([1.0 / counts[group] for group in group_ids], dtype=np.float64)

    @staticmethod
    def _loader(
        torch,
        arrays: Tuple[np.ndarray, ...],
        batch_size: int,
        seed: int,
        sample_weights: Optional[np.ndarray],
        pin_memory: bool,
    ):
        tensors = []
        for values in arrays:
            if np.issubdtype(values.dtype, np.integer):
                tensors.append(torch.from_numpy(values.astype(np.int64, copy=False)))
            else:
                tensors.append(torch.from_numpy(values.astype(np.float32, copy=False)))
        dataset = torch.utils.data.TensorDataset(*tensors)
        generator = torch.Generator().manual_seed(seed)
        if sample_weights is None:
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                generator=generator,
                pin_memory=pin_memory,
            )
        sampler = torch.utils.data.WeightedRandomSampler(
            torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(dataset),
            replacement=True,
            generator=generator,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            pin_memory=pin_memory,
        )

    def _pretrain_epoch(
        self,
        torch,
        model,
        optimizer,
        gradient_scaler,
        ssl_loader,
        device,
        amp_enabled: bool,
        stage_name: str,
        epoch: int,
        epoch_count: int,
        total_steps: int,
    ) -> Dict[str, float]:
        model.train()
        model.target_encoder.eval()
        loss_names = ["total", "jepa", "variance"]
        if self.config.losses.covariance > 0:
            loss_names.append("covariance")
        totals = {name: 0.0 for name in loss_names}
        batch_count = len(ssl_loader)
        for batch_index, (values,) in enumerate(ssl_loader, start=1):
            values = values.to(device, non_blocking=True)
            step_index = (epoch - 1) * batch_count + (batch_index - 1)
            self._set_optimizer_learning_rate(
                optimizer,
                self._learning_rate_for_step(step_index, total_steps),
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                jepa, variance, covariance = self._jepa_terms(
                    torch,
                    model,
                    values,
                    include_covariance=self.config.losses.covariance > 0,
                )
                loss = (
                    self.config.losses.jepa * jepa
                    + self.config.losses.variance * variance
                    + self.config.losses.covariance * covariance
                )
            self._optimizer_step(torch, model, optimizer, gradient_scaler, loss)
            batch = len(values)
            batch_loss_names = ("total", "jepa", "variance")
            loss_tensors = [loss, jepa, variance]
            if self.config.losses.covariance > 0:
                batch_loss_names += ("covariance",)
                loss_tensors.append(covariance)
            loss_values = (
                torch.stack(tuple(loss_tensors))
                .detach()
                .float()
                .cpu()
                .tolist()
            )
            batch_losses = dict(zip(batch_loss_names, loss_values))
            for name, value in batch_losses.items():
                totals[name] += value * batch
            _record_batch_losses(
                {
                    "schema_version": 1,
                    "stage": stage_name,
                    "phase": "pretrain",
                    "epoch": epoch,
                    "epoch_count": epoch_count,
                    "batch": batch_index,
                    "batch_count": batch_count,
                    "batch_size": batch,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "losses": batch_losses,
                }
            )
        _flush_batch_loss_log()
        return {name: value / len(ssl_loader.dataset) for name, value in totals.items()}

    def _joint_epoch(
        self,
        torch,
        model,
        optimizer,
        gradient_scaler,
        loaders,
        classification_context,
        source_weights: np.ndarray,
        device,
        amp_enabled: bool,
        stage_name: str,
        epoch: int,
        epoch_count: int,
        total_steps: int,
    ) -> Dict[str, float]:
        model.train()
        model.target_encoder.eval()
        iterators = {name: _cycle(loader) for name, loader in loaders.items()}
        steps = max(len(loader) for loader in loaders.values())
        weights = torch.as_tensor(source_weights, dtype=torch.float32, device=device)
        loss_names = ["total", "jepa", "classification", "variance"]
        if self.config.losses.alignment > 0:
            loss_names.append("alignment")
        if self.config.losses.consistency > 0:
            loss_names.append("consistency")
        if self.config.losses.covariance > 0:
            loss_names.append("covariance")
        totals = {name: 0.0 for name in loss_names}
        for batch_index in range(1, steps + 1):
            step_index = (epoch - 1) * steps + (batch_index - 1)
            self._set_optimizer_learning_rate(
                optimizer,
                self._learning_rate_for_step(step_index, total_steps),
            )
            ssl_values = next(iterators["ssl"])[0].to(device, non_blocking=True)
            class_values, class_labels, class_projects, class_indices = next(
                iterators["classification"]
            )
            class_values = class_values.to(device, non_blocking=True)
            class_labels = class_labels.to(device, non_blocking=True)
            class_projects = class_projects.to(device, non_blocking=True)
            class_indices = class_indices.to(device, non_blocking=True)
            real_values = real_projects = None
            if self.config.losses.alignment > 0:
                real_values, real_projects = next(iterators["real"])
                real_values = real_values.to(device, non_blocking=True)
                real_projects = real_projects.to(device, non_blocking=True)
            target_values = None
            if (
                self.config.losses.alignment > 0
                or self.config.losses.consistency > 0
            ):
                target_values = next(iterators["target"])[0].to(
                    device, non_blocking=True
                )
            step_classification_context = self._refresh_classification_context(
                model, classification_context
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                jepa, variance, covariance = self._jepa_terms(
                    torch,
                    model,
                    ssl_values,
                    include_covariance=self.config.losses.covariance > 0,
                )
                logits = self._classification_logits(
                    model,
                    class_values,
                    step_classification_context,
                    class_indices,
                )
                row_losses = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, class_labels, reduction="none"
                )
                class_weights = weights[class_projects]
                classification = (row_losses * class_weights).sum() / class_weights.sum().clamp_min(1e-8)
                alignment = jepa.new_zeros(())
                if self.config.losses.alignment > 0:
                    source_latent = model.context_latent(real_values)
                    target_latent = model.context_latent(target_values)
                    alignment = weighted_coral_loss(
                        torch, source_latent, target_latent, weights[real_projects]
                    )
                consistency = jepa.new_zeros(())
                if self.config.losses.consistency > 0:
                    first_view = target_values + torch.randn_like(target_values) * self.config.views.consistency_noise_std
                    second_view = target_values + torch.randn_like(target_values) * self.config.views.consistency_noise_std
                    first_probability = torch.sigmoid(
                        self._classification_logits(
                            model, first_view, step_classification_context, None
                        )
                    )
                    second_probability = torch.sigmoid(
                        self._classification_logits(
                            model, second_view, step_classification_context, None
                        )
                    )
                    consistency = torch.nn.functional.mse_loss(
                        first_probability, second_probability
                    )
                loss = (
                    self.config.losses.jepa * jepa
                    + self.config.losses.classification * classification
                    + self.config.losses.alignment * alignment
                    + self.config.losses.consistency * consistency
                    + self.config.losses.variance * variance
                    + self.config.losses.covariance * covariance
                )
            self._optimizer_step(torch, model, optimizer, gradient_scaler, loss)
            loss_by_name = {
                "total": loss,
                "jepa": jepa,
                "classification": classification,
                "variance": variance,
                "alignment": alignment,
                "consistency": consistency,
                "covariance": covariance,
            }
            loss_values = (
                torch.stack(tuple(loss_by_name[name] for name in loss_names))
                .detach()
                .float()
                .cpu()
                .tolist()
            )
            batch_losses = dict(zip(loss_names, loss_values))
            for name, value in batch_losses.items():
                totals[name] += value
            _record_batch_losses(
                {
                    "schema_version": 1,
                    "stage": stage_name,
                    "phase": "joint",
                    "epoch": epoch,
                    "epoch_count": epoch_count,
                    "batch": batch_index,
                    "batch_count": steps,
                    "batch_size": int(len(class_values)),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "losses": batch_losses,
                }
            )
        _flush_batch_loss_log()
        return {name: value / steps for name, value in totals.items()}

    def _learning_rate_for_step(self, step: int, total_steps: int) -> float:
        optimization = self.config.optimization
        if optimization.learning_rate_schedule == "constant":
            return float(optimization.learning_rate)
        if total_steps <= 1:
            return float(optimization.learning_rate_peak)
        warmup_steps = max(
            1,
            min(
                total_steps - 1,
                int(round(total_steps * optimization.learning_rate_warmup_fraction)),
            ),
        )
        if step < warmup_steps:
            progress = float(step) / float(max(warmup_steps - 1, 1))
            return float(
                optimization.learning_rate_warmup_start
                + progress
                * (
                    optimization.learning_rate_peak
                    - optimization.learning_rate_warmup_start
                )
            )
        decay_steps = total_steps - warmup_steps
        decay_progress = float(step - warmup_steps) / float(max(decay_steps - 1, 1))
        decay_progress = min(max(decay_progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        return float(
            optimization.learning_rate_min
            + cosine
            * (optimization.learning_rate_peak - optimization.learning_rate_min)
        )

    @staticmethod
    def _set_optimizer_learning_rate(optimizer, learning_rate: float) -> None:
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = float(learning_rate)

    def _jepa_terms(self, torch, model, values, include_covariance: bool = True):
        mask = sample_feature_mask(
            torch,
            values,
            self.config.views.minimum_mask_ratio,
            self.config.views.maximum_mask_ratio,
        )
        context_view = values.masked_fill(mask, 0.0)
        target_view = values + torch.randn_like(values) * self.config.views.target_noise_std
        context, prediction = model.predicted_target(context_view, mask)
        with torch.no_grad():
            target = model.target_latent(target_view)
        jepa = latent_prediction_loss(torch, prediction, target)
        variance, covariance = variance_covariance_losses(
            torch, context, include_covariance=include_covariance
        )
        return jepa, variance, covariance

    def _optimizer_step(
        self, torch, model, optimizer, gradient_scaler, loss
    ) -> None:
        gradient_scaler.scale(loss).backward()
        gradient_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            self.config.optimization.gradient_clip_norm,
        )
        gradient_scaler.step(optimizer)
        gradient_scaler.update()
        update_ema_target(torch, model, self.config.optimization.ema_momentum)

    def _source_project_weights(self, torch, model, arrays: _TrainingArrays, device):
        was_training = model.training
        model.eval()
        with torch.no_grad():
            target_latent = self._latent_array(torch, model, arrays.target_values, device)
            source_latents = [
                self._latent_array(torch, model, values, device)
                for values in arrays.real_values_by_project
            ]
        if was_training:
            model.train()
            model.target_encoder.eval()
        target_mean, target_covariance = self._moments(target_latent)
        distances = []
        for latent in source_latents:
            source_mean, source_covariance = self._moments(latent)
            mean_distance = float(np.mean((source_mean - target_mean) ** 2))
            covariance_distance = float(
                np.mean((source_covariance - target_covariance) ** 2)
            )
            distances.append(mean_distance + covariance_distance)
        logits = -np.asarray(distances, dtype=np.float64) / self.config.optimization.source_weight_temperature
        logits -= logits.max()
        weights = np.exp(logits)
        return weights / weights.sum()

    @staticmethod
    def _moments(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean = values.mean(axis=0)
        centered = values - mean
        covariance = centered.T @ centered / max(1, len(values) - 1)
        return mean, covariance

    def _latent_array(self, torch, model, values: np.ndarray, device) -> np.ndarray:
        outputs = []
        batch_size = self.config.optimization.batch_size
        for start in range(0, len(values), batch_size):
            tensor = torch.from_numpy(values[start : start + batch_size]).to(device)
            outputs.append(model.context_latent(tensor).cpu().numpy())
        return np.vstack(outputs).astype(np.float64)

    def _validation_loss(
        self,
        torch,
        model,
        values,
        labels,
        classification_context,
        device,
    ) -> float:
        model.eval()
        classification_context = self._refresh_classification_context(
            model, classification_context
        )
        total_loss = 0.0
        batch_size = self.config.optimization.batch_size
        with torch.no_grad():
            for start in range(0, len(values), batch_size):
                tensor = torch.from_numpy(values[start : start + batch_size]).to(device)
                targets = torch.from_numpy(labels[start : start + batch_size]).to(device)
                logits = self._classification_logits(
                    model, tensor, classification_context, None
                )
                total_loss += float(
                    torch.nn.functional.binary_cross_entropy_with_logits(
                        logits, targets, reduction="sum"
                    ).cpu()
                )
        return total_loss / len(values)

    def _scores(
        self,
        torch,
        model,
        values: np.ndarray,
        arrays: _TrainingArrays,
        device,
    ) -> np.ndarray:
        model.eval()
        classification_context = self._classification_context(torch, arrays, device)
        classification_context = self._refresh_classification_context(
            model, classification_context
        )
        outputs = []
        batch_size = self.config.optimization.batch_size
        with torch.no_grad():
            for start in range(0, len(values), batch_size):
                tensor = torch.from_numpy(values[start : start + batch_size]).to(device)
                logits = self._classification_logits(
                    model, tensor, classification_context, None
                )
                outputs.append(torch.sigmoid(logits).cpu())
        return torch.cat(outputs).numpy().astype(np.float64)

    def _log_epoch(
        self,
        stage_name: str,
        phase: str,
        epoch: int,
        total_epochs: int,
        losses: Mapping[str, float],
    ) -> None:
        interval = self.config.optimization.log_every_epochs
        if epoch != 1 and epoch != total_epochs and epoch % interval != 0:
            return
        loss_text = " ".join(
            f"{name}={value:.5f}" for name, value in losses.items()
        )
        print(
            f"PROGRESS {self.progress_name} stage={stage_name} phase={phase} "
            f"epoch={epoch}/{total_epochs} {loss_text}",
            flush=True,
        )

    def _build_model(self, torch, feature_dim: int):
        raise NotImplementedError

    @staticmethod
    def _classification_context(torch, arrays: _TrainingArrays, device):
        return None

    @staticmethod
    def _refresh_classification_context(model, classification_context):
        return classification_context

    @staticmethod
    def _classification_logits(model, values, classification_context, query_indices):
        return model.classification_logits(values)

    @staticmethod
    def _classifier_name() -> str:
        return "linear"

    @staticmethod
    def _classification_context_name() -> str:
        return "none"

    @staticmethod
    def _architecture_metadata() -> Dict[str, str]:
        raise NotImplementedError

    @staticmethod
    def _seed_torch(torch, seed: int) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


class KANJEPAPredictor(JEPAPredictorBase):
    """Transductive KAN-JEPA predictor with source-only supervised validation."""

    model_name = "KAN-JEPA"
    metadata_name = "kan-jepa"
    progress_name = "kan_jepa"

    def __init__(self, config: KANJEPAConfig) -> None:
        super().__init__(config)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "KANJEPAPredictor":
        return cls(KANJEPAConfig.from_mapping(values))

    def _build_model(self, torch, feature_dim: int):
        return build_kan_jepa_model(
            torch,
            feature_dim=feature_dim,
            encoder_config=self.config.encoder,
            predictor_hidden_dims=self.config.jepa_predictor.hidden_dims,
            mask_embedding_dim=self.config.jepa_predictor.mask_embedding_dim,
            predictor_dropout=self.config.jepa_predictor.dropout,
            classifier_config=self.config.classifier,
        )

    @staticmethod
    def _classification_context(torch, arrays: _TrainingArrays, device):
        return (
            torch.from_numpy(arrays.augmented_values).to(device),
            torch.from_numpy(arrays.augmented_labels.astype(np.int64)).to(device),
        )

    @staticmethod
    def _refresh_classification_context(model, classification_context):
        candidate_values, candidate_labels = classification_context[:2]
        return (
            candidate_values,
            candidate_labels,
            model.classification_candidate_keys(candidate_values),
        )

    @staticmethod
    def _classification_logits(model, values, classification_context, query_indices):
        candidate_values, candidate_labels, candidate_keys = classification_context
        return model.classification_logits(
            values,
            candidate_values,
            candidate_labels,
            query_indices,
            candidate_keys,
        )

    @staticmethod
    def _classifier_name() -> str:
        return "TabR"

    @staticmethod
    def _classification_context_name() -> str:
        return "labeled augmented-source only; training query self-excluded"

    @staticmethod
    def _architecture_metadata() -> Dict[str, str]:
        return {
            "context_encoder": "Chebyshev-KAN",
            "target_encoder": "EMA-Chebyshev-KAN",
            "latent_predictor": "MLP",
        }
