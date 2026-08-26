import json
from pathlib import Path

import pandas as pd

from cpdp.oversampling_seed_search.oracle import (
    _successful_payload,
    load_oracle_seed_search_config,
    rank_oracle_candidates,
)


def test_oracle_config_generates_one_hundred_seeds(tmp_path: Path) -> None:
    config_path = tmp_path / "oracle.yaml"
    config_path.write_text(
        """
schema_version: 1
paths:
  experiment_config: experiment.yaml
  output_dir: output
search:
  candidate_seed_start: 1001
  candidate_seed_count: 100
  predictor_seed: 77
selection:
  metrics: [auc, mcc]
storage:
  retain_candidate_caches: false
""".strip(),
        encoding="utf-8",
    )

    config = load_oracle_seed_search_config(config_path)

    assert len(config.candidate_seeds) == 100
    assert config.candidate_seeds[0] == 1001
    assert config.candidate_seeds[-1] == 1100
    assert config.predictor_seed == 77


def test_oracle_ranking_uses_equal_weight_metric_ranks() -> None:
    metrics = pd.DataFrame(
        [
            {"candidate_seed": 1, "auc": 0.90, "mcc": 0.30},
            {"candidate_seed": 2, "auc": 0.80, "mcc": 0.70},
            {"candidate_seed": 3, "auc": 0.70, "mcc": 0.20},
        ]
    )

    ranking = rank_oracle_candidates(metrics, ("auc", "mcc"))

    assert list(ranking["candidate_seed"]) == [2, 1, 3]
    assert ranking.iloc[0]["mean_metric_rank"] < ranking.iloc[2]["mean_metric_rank"]


def test_oracle_resume_requires_matching_signature(tmp_path: Path) -> None:
    path = tmp_path / "seed.json"
    path.write_text(
        json.dumps(
            {
                "status": "success",
                "search_signature": "current",
                "target_labels_used_for_seed_selection": True,
            }
        ),
        encoding="utf-8",
    )

    assert _successful_payload(path, "current") is not None
    assert _successful_payload(path, "changed") is None
