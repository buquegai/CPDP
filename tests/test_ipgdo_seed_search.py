from pathlib import Path

import numpy as np
import pandas as pd

from cpdp.oversampling.types import AugmentationResult
from cpdp.oversampling_seed_search.cache import (
    load_augmentation_cache,
    save_augmentation_cache,
)
from cpdp.oversampling_seed_search.search import rank_target_seeds
from cpdp.types import LabeledProject, ProjectFeatures


def _project(name: str, synthetic: bool = False) -> LabeledProject:
    values = np.asarray(((1.0, 2.0), (2.0, 3.0), (3.0, 4.0)), dtype=np.float64)
    labels = np.asarray((0, 1, 1), dtype=np.int64)
    row_ids = np.asarray((0, 1, -1 if synthetic else 2), dtype=np.int64)
    return LabeledProject(
        ProjectFeatures(
            family="TEST",
            name=name,
            values=values,
            feature_names=("loc", "wmc"),
            effort=values[:, 0],
            effort_name="loc",
            row_ids=row_ids,
        ),
        labels,
    )


def test_augmentation_cache_rejects_changed_configuration(tmp_path: Path) -> None:
    source = _project("source")
    augmented = _project("source", synthetic=True)
    cache = tmp_path / "seed_7.npz"
    save_augmentation_cache(
        cache,
        AugmentationResult((augmented,), {"name": "IPGDO"}),
        target_key="TEST/target",
        oversampling_seed=7,
        oversampling_config={"target_minority_ratio": 0.6},
        sources=(source,),
    )

    restored = load_augmentation_cache(
        cache,
        target_key="TEST/target",
        oversampling_seed=7,
        oversampling_config={"target_minority_ratio": 0.6},
        sources=(source,),
    )

    np.testing.assert_array_equal(restored.projects[0].features.row_ids, (0, 1, -1))
    assert restored.metadata["cache"]["reused"] is True

    try:
        load_augmentation_cache(
            cache,
            target_key="TEST/target",
            oversampling_seed=7,
            oversampling_config={"target_minority_ratio": 1.0},
            sources=(source,),
        )
    except ValueError as exc:
        assert "oversampling_config_signature changed" in str(exc)
    else:
        raise AssertionError("Changed IPGDO parameters must invalidate the cache")


def test_seed_ranking_uses_project_metric_ranks_before_quality() -> None:
    metric_rows = pd.DataFrame(
        [
            {"candidate_seed": 1, "source_project": "TEST/a", "auc": 0.9, "mcc": 0.8},
            {"candidate_seed": 2, "source_project": "TEST/a", "auc": 0.7, "mcc": 0.6},
            {"candidate_seed": 1, "source_project": "TEST/b", "auc": 0.8, "mcc": 0.7},
            {"candidate_seed": 2, "source_project": "TEST/b", "auc": 0.6, "mcc": 0.5},
        ]
    )
    quality_rows = pd.DataFrame(
        [
            {"candidate_seed": 1, "defect_manifold_in_range_rate": 0.5},
            {"candidate_seed": 2, "defect_manifold_in_range_rate": 1.0},
        ]
    )

    ranking = rank_target_seeds(metric_rows, quality_rows, ("auc", "mcc"))

    assert list(ranking["candidate_seed"]) == [1, 2]
    assert ranking.iloc[0]["mean_utility_rank"] < ranking.iloc[1]["mean_utility_rank"]
