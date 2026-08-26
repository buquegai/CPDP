from __future__ import annotations

import json

import numpy as np
import pytest

from cpdp.oversampling.config import OversamplingConfig
from cpdp.oversampling.pipeline import IPGDOOversampler
from cpdp.types import LabeledProject, ProjectFeatures


def _source_project(name: str = "source") -> LabeledProject:
    values = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.1, 0.3],
            [0.4, 0.3, 0.2],
            [0.6, 0.4, 0.5],
            [1.0, 1.0, 1.2],
            [1.4, 1.2, 1.1],
        ],
        dtype=np.float64,
    )
    labels = np.asarray([0, 0, 0, 0, 1, 1], dtype=np.int64)
    features = ProjectFeatures(
        family="TEST",
        name=name,
        values=values,
        feature_names=("loc", "wmc", "cbo"),
        effort=values[:, 0],
        effort_name="loc",
        row_ids=np.arange(len(values)),
    )
    return LabeledProject(features=features, labels=labels)


class _StubDiffusion:
    sample_sizes: list[int] = []
    fitted_values: np.ndarray | None = None
    fitted_project_ids: np.ndarray | None = None
    fitted_formula_guidance = None

    def __init__(self, *args, **kwargs) -> None:
        self.input_dim = int(kwargs["input_dim"])

    def fit(self, normalized_defects, project_ids, seed, **kwargs):
        self.__class__.fitted_values = np.asarray(normalized_defects)
        self.__class__.fitted_project_ids = np.asarray(project_ids)
        self.__class__.fitted_formula_guidance = kwargs.get("formula_guidance")
        return {
            "training_data": "real_source_defects_only",
            "objective": "ddpm_noise_mse_plus_frozen_formula_guidance",
        }

    def sample_defects(self, sample_count, project_id):
        self.sample_sizes.append(int(sample_count))
        return np.zeros((sample_count, self.input_dim), dtype=np.float64)


def _config() -> OversamplingConfig:
    return OversamplingConfig.from_mapping(
        {
            "target_minority_ratio": 1.0,
            "diffusion": {"epochs": 1, "timesteps": 2},
            "formula": {"minimum_reliability": 0.0},
        }
    )


def test_ipgdo_generates_exact_quota_without_selection(monkeypatch) -> None:
    import cpdp.oversampling.pipeline as pipeline_module

    _StubDiffusion.sample_sizes = []
    monkeypatch.setattr(
        pipeline_module, "ProjectConditionedTabularDDPM", _StubDiffusion
    )

    result = IPGDOOversampler(_config()).fit_resample(
        (_source_project(),), seed=11
    )

    assert _StubDiffusion.sample_sizes == [2]
    assert result.metadata["total_synthetic_selected"] == 2
    assert result.projects[0].features.n_samples == 8
    assert result.metadata["selection"]["enabled"] is False
    assert result.metadata["formula_guidance"]["enabled"] is True
    assert _StubDiffusion.fitted_formula_guidance is not None
    serialized = json.dumps(result.metadata, sort_keys=True).lower()
    assert "knn" not in serialized


def test_ipgdo_formula_discovery_uses_source_rows_only(monkeypatch) -> None:
    import cpdp.oversampling.pipeline as pipeline_module

    _StubDiffusion.fitted_values = None
    _StubDiffusion.fitted_project_ids = None
    monkeypatch.setattr(
        pipeline_module, "ProjectConditionedTabularDDPM", _StubDiffusion
    )

    sources = (_source_project("source_a"), _source_project("source_b"))
    result = IPGDOOversampler(_config()).fit_resample(sources, seed=11)

    assert _StubDiffusion.fitted_values is not None
    assert _StubDiffusion.fitted_project_ids is not None
    assert len(_StubDiffusion.fitted_values) == 4
    assert result.metadata["formula_guidance"]["source_only"] is True


def test_ipgdo_rejects_removed_knn_configuration() -> None:
    with pytest.raises(ValueError, match="Unsupported oversampling option"):
        OversamplingConfig.from_mapping({"knn": {"neighbors": 5}})
