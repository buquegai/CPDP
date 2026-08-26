from pathlib import Path

import numpy as np
import pytest

from cpdp.config import load_config
from cpdp.predictors.jepa_losses import latent_prediction_loss
from cpdp.predictors.kan import update_ema_target
from cpdp.predictors.transformer import (
    TransformerEncoderConfig,
    TransformerPredictorConfig,
    build_transformer_jepa_model,
)
from cpdp.predictors.transformer_jepa import TransformerJEPAPredictor
from cpdp.runner import _method_uses_exclusive_cuda
from cpdp.types import LabeledProject, ProjectFeatures


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _project(name: str, offset: float, synthetic_rows: int = 0) -> LabeledProject:
    random = np.random.default_rng(sum(ord(character) for character in name))
    real_values = random.normal(offset, 0.7, size=(24, 4))
    real_labels = (
        real_values[:, 0] + real_values[:, 1] - real_values[:, 2] > 2.0 * offset
    ).astype(np.int64)
    if len(np.unique(real_labels)) < 2:
        real_labels[:12] = 0
        real_labels[12:] = 1
    synthetic = random.normal(offset + 0.4, 0.5, size=(synthetic_rows, 4))
    values = np.vstack((real_values, synthetic))
    labels = np.concatenate((real_labels, np.ones(synthetic_rows, dtype=np.int64)))
    row_ids = np.concatenate(
        (
            np.arange(len(real_values), dtype=np.int64),
            -np.arange(1, synthetic_rows + 1, dtype=np.int64),
        )
    )
    features = ProjectFeatures(
        family="toy",
        name=name,
        values=values,
        feature_names=("loc", "wmc", "cbo", "rfc"),
        effort=values[:, 0],
        effort_name="loc",
        row_ids=row_ids,
    )
    return LabeledProject(features=features, labels=labels)


def _smoke_config():
    return {
        "device": "cpu",
        "encoder": {
            "model_dim": 16,
            "latent_dim": 8,
            "num_heads": 4,
            "num_layers": 1,
            "feedforward_dim": 24,
            "dropout": 0.0,
        },
        "jepa_predictor": {
            "model_dim": 16,
            "num_heads": 4,
            "num_layers": 1,
            "feedforward_dim": 24,
            "dropout": 0.0,
        },
        "optimization": {
            "pretrain_epochs": 1,
            "joint_epochs": 2,
            "batch_size": 64,
            "early_stopping_patience": 1,
            "source_weight_update_interval": 1,
            "refit_all_sources": True,
            "log_every_epochs": 10,
            "use_amp": False,
        },
    }


def test_transformer_model_shapes_gradients_and_ema() -> None:
    torch = pytest.importorskip("torch")
    model = build_transformer_jepa_model(
        torch,
        feature_dim=4,
        encoder_config=TransformerEncoderConfig(
            model_dim=16,
            latent_dim=8,
            num_heads=4,
            num_layers=1,
            feedforward_dim=24,
            dropout=0.0,
        ),
        predictor_config=TransformerPredictorConfig(
            model_dim=16,
            num_heads=4,
            num_layers=1,
            feedforward_dim=24,
            dropout=0.0,
        ),
    )
    values = torch.randn(10, 4)
    mask = torch.rand(10, 4) > 0.6
    context, prediction = model.predicted_target(values.masked_fill(mask, 0.0), mask)
    target = model.target_latent(values).detach()
    loss = latent_prediction_loss(torch, prediction, target)
    loss.backward()

    assert context.shape == (10, 8)
    assert prediction.shape == (10, 8)
    assert target.shape == (10, 8)
    assert isinstance(model.context_encoder.transformer, torch.nn.TransformerEncoder)
    assert isinstance(model.target_encoder.transformer, torch.nn.TransformerEncoder)
    assert isinstance(model.predictor.transformer, torch.nn.TransformerEncoder)
    assert any(
        parameter.grad is not None
        for parameter in model.context_encoder.parameters()
    )
    assert any(parameter.grad is not None for parameter in model.predictor.parameters())
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())

    before = next(model.target_encoder.parameters()).detach().clone()
    with torch.no_grad():
        next(model.context_encoder.parameters()).add_(1.0)
    update_ema_target(torch, model, momentum=0.5)
    after = next(model.target_encoder.parameters()).detach()
    assert not torch.equal(before, after)


def test_transformer_predictor_preserves_data_contract() -> None:
    pytest.importorskip("torch")
    source_a = _project("source-a", 0.0, synthetic_rows=5)
    source_b = _project("source-b", 0.3, synthetic_rows=7)
    target = _project("target", 0.15).without_labels()
    predictor = TransformerJEPAPredictor.from_mapping(_smoke_config())

    result = predictor.fit_predict((source_a, source_b), target, seed=12345)

    assert result.scores.shape == (target.features.n_samples,)
    assert np.isfinite(result.scores).all()
    assert 0.0 <= result.threshold <= 1.0
    assert result.metadata["predictor"] == "transformer-jepa"
    assert result.metadata["architecture"]["context_encoder"] == "Transformer"
    assert result.metadata["architecture"]["target_encoder"] == "EMA-Transformer"
    assert result.metadata["architecture"]["latent_predictor"] == "Transformer"
    assert result.metadata["data_contract"]["synthetic_rows"] == 12
    assert result.metadata["data_contract"]["target_labels_used"] is False
    assert result.metadata["data_contract"]["pseudo_labels_used"] is False
    assert result.metadata["final_training"]["real_source_rows"] == 48
    assert result.metadata["final_training"]["augmented_source_rows"] == 60


@pytest.mark.parametrize(
    "method_name",
    (
        "ipgdo_kan_jepa",
        "ipgdo_transformer_jepa",
    ),
)
def test_proposed_jepa_methods_use_exclusive_cuda_when_available(
    method_name: str,
) -> None:
    torch = pytest.importorskip("torch")
    config = load_config(PROJECT_ROOT / "configs" / "experiment.yaml")

    assert _method_uses_exclusive_cuda(config, "full", method_name) is bool(
        torch.cuda.is_available()
    )
