import numpy as np
import pytest

from cpdp.predictors.jepa_losses import (
    latent_prediction_loss,
    variance_covariance_losses,
    weighted_coral_loss,
)
from cpdp.predictors.kan import (
    KANEncoderConfig,
    build_kan_jepa_model,
    update_ema_target,
)
from cpdp.predictors.kan_jepa import KANJEPAPredictor
from cpdp.predictors.tabr import TabRClassifierConfig
from cpdp.types import LabeledProject, ProjectFeatures


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
    labels = np.concatenate(
        (real_labels, np.ones(synthetic_rows, dtype=np.int64))
    )
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
            "hidden_dims": [12],
            "latent_dim": 6,
            "grid_size": 3,
            "spline_order": 2,
            "dropout": 0.0,
        },
        "jepa_predictor": {
            "hidden_dims": [12],
            "mask_embedding_dim": 4,
            "dropout": 0.0,
        },
        "classifier": {
            "context_size": 4,
            "hidden_dim": 12,
            "hidden_multiplier": 2.0,
            "encoder_blocks": 1,
            "predictor_blocks": 1,
            "context_dropout": 0.0,
            "dropout": 0.0,
            "candidate_batch_size": 64,
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


def test_kan_model_shapes_gradients_and_ema() -> None:
    torch = pytest.importorskip("torch")
    model = build_kan_jepa_model(
        torch,
        feature_dim=4,
        encoder_config=KANEncoderConfig(
            hidden_dims=(12,), latent_dim=6, grid_size=3, spline_order=2
        ),
        predictor_hidden_dims=(12,),
        mask_embedding_dim=4,
        predictor_dropout=0.0,
        classifier_config=TabRClassifierConfig(
            context_size=4,
            hidden_dim=12,
            encoder_blocks=1,
            predictor_blocks=1,
            context_dropout=0.0,
            dropout=0.0,
        ),
    )
    values = torch.randn(10, 4)
    mask = torch.rand(10, 4) > 0.6
    context, prediction = model.predicted_target(values.masked_fill(mask, 0.0), mask)
    target = model.target_latent(values).detach()
    loss = latent_prediction_loss(torch, prediction, target)
    loss.backward()

    assert context.shape == (10, 6)
    assert prediction.shape == (10, 6)
    assert any(parameter.grad is not None for parameter in model.context_encoder.parameters())
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())

    before = next(model.target_encoder.parameters()).detach().clone()
    with torch.no_grad():
        next(model.context_encoder.parameters()).add_(1.0)
    update_ema_target(torch, model, momentum=0.5)
    after = next(model.target_encoder.parameters()).detach()
    assert not torch.equal(before, after)


def test_tabr_classifier_excludes_self_and_backpropagates_to_kan() -> None:
    torch = pytest.importorskip("torch")
    model = build_kan_jepa_model(
        torch,
        feature_dim=4,
        encoder_config=KANEncoderConfig(
            hidden_dims=(12,), latent_dim=6, grid_size=3, spline_order=2, dropout=0.0
        ),
        predictor_hidden_dims=(12,),
        mask_embedding_dim=4,
        predictor_dropout=0.0,
        classifier_config=TabRClassifierConfig(
            context_size=3,
            hidden_dim=12,
            encoder_blocks=1,
            predictor_blocks=1,
            context_dropout=0.0,
            dropout=0.0,
        ),
    )
    candidates = torch.randn(10, 4)
    labels = torch.tensor([0, 1] * 5)
    query_indices = torch.arange(5)

    with torch.no_grad():
        candidate_latent = model.context_latent(candidates)
        _, candidate_keys = model.classifier.encode(candidate_latent)
        context_indices = model.classifier.select_context_indices(
            candidate_keys[:5], candidate_keys, query_indices
        )
    assert all(
        int(query_indices[row]) not in context_indices[row].tolist()
        for row in range(len(query_indices))
    )

    logits = model.classification_logits(
        candidates[:5], candidates, labels, query_indices
    )
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, labels[:5].float()
    )
    loss.backward()

    assert logits.shape == (5,)
    assert any(
        parameter.grad is not None for parameter in model.context_encoder.parameters()
    )
    assert any(parameter.grad is not None for parameter in model.classifier.parameters())


def test_jepa_regularizers_are_finite() -> None:
    torch = pytest.importorskip("torch")
    source = torch.randn(16, 8, requires_grad=True)
    target = torch.randn(12, 8)
    variance, covariance = variance_covariance_losses(torch, source)
    coral = weighted_coral_loss(torch, source, target, torch.ones(16))
    total = variance + covariance + coral
    total.backward()

    assert torch.isfinite(total)
    assert source.grad is not None
    assert torch.isfinite(source.grad).all()


def test_predictor_preserves_real_synthetic_data_contract() -> None:
    pytest.importorskip("torch")
    source_a = _project("source-a", 0.0, synthetic_rows=5)
    source_b = _project("source-b", 0.3, synthetic_rows=7)
    target = _project("target", 0.15).without_labels()
    predictor = KANJEPAPredictor.from_mapping(_smoke_config())

    result = predictor.fit_predict((source_a, source_b), target, seed=12345)

    assert result.scores.shape == (target.features.n_samples,)
    assert np.isfinite(result.scores).all()
    assert 0.0 <= result.threshold <= 1.0
    assert result.metadata["data_contract"]["synthetic_rows"] == 12
    assert result.metadata["data_contract"]["target_labels_used"] is False
    assert result.metadata["data_contract"]["pseudo_labels_used"] is False
    assert result.metadata["architecture"]["classifier"] == "TabR"
    assert result.metadata["final_training"]["real_source_rows"] == 48
    assert result.metadata["final_training"]["augmented_source_rows"] == 60
