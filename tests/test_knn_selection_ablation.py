from __future__ import annotations

import numpy as np

from cpdp.oversampling.manifold_filter import KNNFilterEvaluation
from cpdp.oversampling.types import freeze_array
from cpdp.oversampling_validation.knn_selection_ablation import (
    SELECTION_STRATEGIES,
    KNNSelectionConfig,
    ProjectCandidatePool,
    SharedCandidatePoolBuilder,
    SharedCandidatePools,
    build_strategy_augmentation,
    select_project_candidates,
)
from cpdp.types import LabeledProject, ProjectFeatures


def _project(name: str, offset: float = 0.0) -> LabeledProject:
    values = np.asarray(
        [
            [0.0, 0.0],
            [0.2, 0.1],
            [0.4, 0.3],
            [0.6, 0.4],
            [1.0, 1.0],
            [1.4, 1.2],
        ],
        dtype=np.float64,
    )
    values += offset
    labels = np.asarray([0, 0, 0, 0, 1, 1], dtype=np.int64)
    features = ProjectFeatures(
        family="TEST",
        name=name,
        values=values,
        feature_names=("loc", "wmc"),
        effort=values[:, 0],
        effort_name="loc",
        row_ids=np.arange(len(values)),
    )
    return LabeledProject(features=features, labels=labels)


def _pool() -> ProjectCandidatePool:
    candidates = np.asarray(
        [[10.0, 0.0], [11.0, 0.0], [12.0, 0.0], [13.0, 0.0]],
        dtype=np.float64,
    )
    return ProjectCandidatePool(
        project_key="TEST/source",
        requested=2,
        candidates=freeze_array(candidates, np.float64),
        accepted=freeze_array([False, True, True, True], bool),
        quality_scores=freeze_array([1.0, 0.3, 0.1, 0.2], np.float64),
        mean_distances=freeze_array([9.0, 3.0, 1.0, 2.0], np.float64),
        metadata={},
    )


def test_selection_strategies_use_one_shared_pool() -> None:
    source = _project("source")
    pool = _pool()
    shared = SharedCandidatePools(
        sources=(source,),
        project_pools=(pool,),
        diffusion_metadata={"fit_calls": 1},
        generation_seconds=1.0,
    )

    augmentations = {
        strategy: build_strategy_augmentation(shared, strategy, seed=17)
        for strategy in SELECTION_STRATEGIES
    }

    hashes = {
        result.metadata["shared_candidate_pool_sha256"]
        for result in augmentations.values()
    }
    assert hashes == {shared.candidate_pool_sha256}
    assert all(
        result.metadata["total_synthetic_selected"] == 2
        for result in augmentations.values()
    )


def test_knn_median_strategy_selects_closest_to_internal_median() -> None:
    selected, metadata = select_project_candidates(
        _pool(), "knn_interval_median", seed=17
    )

    np.testing.assert_array_equal(selected[:, 0], [12.0, 13.0])
    assert metadata["eligible"] == 3
    assert metadata["selected"] == 2


class _StubDiffusion:
    fit_calls = 0
    sample_calls = 0

    def __init__(self, input_dim, project_count, config, device) -> None:
        self.input_dim = input_dim

    def fit(self, normalized_defects, project_ids, seed):
        self.__class__.fit_calls += 1
        return {"fit_calls": self.fit_calls, "seed": seed}

    def sample_defects(self, sample_count, project_id):
        self.__class__.sample_calls += 1
        return np.zeros((sample_count, self.input_dim), dtype=np.float64)


class _AcceptAllFilter:
    def __init__(self, real_defects, config) -> None:
        pass

    def evaluate(self, candidates):
        count = len(candidates)
        return KNNFilterEvaluation(
            accepted=np.ones(count, dtype=bool),
            quality_scores=np.arange(count, dtype=np.float64),
            mean_distances=np.ones(count, dtype=np.float64),
            metadata={"candidates": count, "accepted": count, "nonfinite": 0},
        )

    def metadata(self):
        return {"neighbors": 1, "quality_rule": "test", "ranking_rule": "test"}


def test_builder_fits_diffusion_once_for_all_source_projects(monkeypatch) -> None:
    import cpdp.oversampling_validation.knn_selection_ablation as module

    _StubDiffusion.fit_calls = 0
    _StubDiffusion.sample_calls = 0
    monkeypatch.setattr(module, "ProjectConditionedTabularDDPM", _StubDiffusion)
    monkeypatch.setattr(module, "DefectManifoldFilter", _AcceptAllFilter)
    config = KNNSelectionConfig.from_mapping(
        {
            "target_minority_ratio": 1.0,
            "candidate_batch_multiplier": 1.0,
            "max_generation_rounds": 2,
            "diffusion": {"epochs": 1, "timesteps": 2},
            "knn": {"neighbors": 1, "lower_quantile": 0.1, "upper_quantile": 0.9},
        }
    )

    shared = SharedCandidatePoolBuilder(config).build(
        (_project("one"), _project("two", 10.0)), seed=23
    )

    assert _StubDiffusion.fit_calls == 1
    assert _StubDiffusion.sample_calls == 2
    assert len(shared.project_pools) == 2
    assert all(len(pool.candidates) == pool.requested for pool in shared.project_pools)
