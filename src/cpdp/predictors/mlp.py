from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from cpdp.predictors.base import BasePredictor
from cpdp.predictors.threshold import select_mcc_threshold
from cpdp.types import LabeledProject, PredictionResult, UnlabeledProject


@dataclass(frozen=True)
class MLPConfig:
    device: str = "auto"
    hidden_dims: tuple = (128, 64)
    dropout: float = 0.2
    epochs: int = 200
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    validation_fraction: float = 0.15
    early_stopping_patience: int = 20
    class_weighted_loss: bool = False
    gradient_clip_norm: float = 1.0
    use_amp: bool = True

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "MLPConfig":
        config = cls(
            device=str(values.get("device", "auto")),
            hidden_dims=tuple(int(item) for item in values.get("hidden_dims", (128, 64))),
            dropout=float(values.get("dropout", 0.2)),
            epochs=int(values.get("epochs", 200)),
            batch_size=int(values.get("batch_size", 256)),
            learning_rate=float(values.get("learning_rate", 1e-3)),
            weight_decay=float(values.get("weight_decay", 1e-5)),
            validation_fraction=float(values.get("validation_fraction", 0.15)),
            early_stopping_patience=int(values.get("early_stopping_patience", 20)),
            class_weighted_loss=bool(values.get("class_weighted_loss", False)),
            gradient_clip_norm=float(values.get("gradient_clip_norm", 1.0)),
            use_amp=bool(values.get("use_amp", True)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("predictor.device must be 'auto', 'cpu', or 'cuda'")
        if not self.hidden_dims or min(self.hidden_dims) < 1:
            raise ValueError("predictor.hidden_dims must contain positive dimensions")
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError("predictor.dropout must be between 0 and 1")
        if self.epochs < 1 or self.batch_size < 2:
            raise ValueError("predictor epochs and batch_size are too small")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("predictor optimizer values are invalid")
        if not 0.0 <= self.validation_fraction < 0.5:
            raise ValueError("predictor.validation_fraction must be in [0, 0.5)")
        if self.early_stopping_patience < 1:
            raise ValueError("predictor.early_stopping_patience must be positive")
        if self.gradient_clip_norm <= 0:
            raise ValueError("predictor.gradient_clip_norm must be positive")


def _resolve_device(torch, requested: str):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for the MLP, but it is unavailable")
    return torch.device(requested)


def _build_network(torch, input_dim: int, config: MLPConfig):
    layers = []
    previous = input_dim
    for hidden in config.hidden_dims:
        layers.extend(
            (
                torch.nn.Linear(previous, hidden),
                torch.nn.LayerNorm(hidden),
                torch.nn.SiLU(),
                torch.nn.Dropout(config.dropout),
            )
        )
        previous = hidden
    layers.append(torch.nn.Linear(previous, 1))
    return torch.nn.Sequential(*layers)


class MLPPredictor(BasePredictor):
    """Temporary source-supervised predictor replaceable without DDPM changes."""

    def __init__(self, config: MLPConfig) -> None:
        self.config = config

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "MLPPredictor":
        return cls(MLPConfig.from_mapping(values))

    def fit_predict(
        self,
        sources: Sequence[LabeledProject],
        target: UnlabeledProject,
        seed: int,
    ) -> PredictionResult:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("The temporary MLP predictor requires PyTorch") from exc
        if not sources:
            raise ValueError("MLP predictor requires at least one source project")
        values = np.vstack([source.features.values for source in sources])
        labels = np.concatenate([source.labels for source in sources]).astype(np.int64)
        indices = np.arange(len(labels))
        use_validation = (
            self.config.validation_fraction > 0
            and int(np.bincount(labels, minlength=2).min()) >= 2
        )
        if use_validation:
            train_indices, validation_indices = train_test_split(
                indices,
                test_size=self.config.validation_fraction,
                random_state=int(seed % (2**32 - 1)),
                stratify=labels,
            )
        else:
            train_indices = indices
            validation_indices = np.empty(0, dtype=np.int64)
        scaler = StandardScaler().fit(values[train_indices])
        train_values = scaler.transform(values[train_indices]).astype(np.float32)
        train_labels = labels[train_indices].astype(np.float32)
        validation_values = scaler.transform(values[validation_indices]).astype(np.float32)
        target_values = scaler.transform(target.features.values).astype(np.float32)

        device = _resolve_device(torch, self.config.device)
        network = _build_network(torch, values.shape[1], self.config).to(device)
        optimizer = torch.optim.AdamW(
            network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        if self.config.class_weighted_loss:
            counts = np.bincount(train_labels.astype(np.int64), minlength=2)
            positive_weight = torch.tensor(
                [counts[0] / max(counts[1], 1)], dtype=torch.float32, device=device
            )
        else:
            positive_weight = None
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weight)
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(train_values), torch.from_numpy(train_labels)
        )
        generator = torch.Generator().manual_seed(int(seed))
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            generator=generator,
        )
        amp_enabled = bool(self.config.use_amp and device.type == "cuda")
        gradient_scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        best_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0
        completed_epochs = 0

        for epoch in range(self.config.epochs):
            network.train()
            for batch_values, batch_labels in loader:
                batch_values = batch_values.to(device, non_blocking=True)
                batch_labels = batch_labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=amp_enabled):
                    logits = network(batch_values).squeeze(1)
                    loss = criterion(logits, batch_labels)
                gradient_scaler.scale(loss).backward()
                gradient_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    network.parameters(), self.config.gradient_clip_norm
                )
                gradient_scaler.step(optimizer)
                gradient_scaler.update()
            completed_epochs = epoch + 1
            if len(validation_indices) == 0:
                continue
            validation_loss = self._loss(
                torch, network, validation_values, labels[validation_indices], device, criterion
            )
            if validation_loss < best_loss - 1e-6:
                best_loss = validation_loss
                best_state = {
                    name: parameter.detach().cpu().clone()
                    for name, parameter in network.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.config.early_stopping_patience:
                    break
        if best_state is not None:
            network.load_state_dict(best_state)
        if len(validation_indices):
            validation_scores = self._scores(torch, network, validation_values, device)
            threshold = select_mcc_threshold(
                labels[validation_indices], validation_scores
            )
        else:
            threshold = 0.5
        scores = self._scores(torch, network, target_values, device)
        return PredictionResult(
            scores=scores,
            threshold=threshold,
            metadata={
                "predictor": "temporary-mlp",
                "device": str(device),
                "amp_enabled": amp_enabled,
                "configuration": asdict(self.config),
                "training_rows": int(len(train_indices)),
                "validation_rows": int(len(validation_indices)),
                "completed_epochs": completed_epochs,
                "best_validation_loss": None if best_state is None else best_loss,
                "threshold_policy": (
                    "source-validation-mcc" if len(validation_indices) else "fixed-0.5"
                ),
            },
        )

    @staticmethod
    def _scores(torch, network, values: np.ndarray, device) -> np.ndarray:
        network.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(values).to(device)
            return torch.sigmoid(network(tensor).squeeze(1)).cpu().numpy().astype(np.float64)

    @staticmethod
    def _loss(torch, network, values, labels, device, criterion) -> float:
        network.eval()
        with torch.no_grad():
            logits = network(torch.from_numpy(values).to(device)).squeeze(1)
            target = torch.as_tensor(labels, dtype=torch.float32, device=device)
            return float(criterion(logits, target).cpu())
