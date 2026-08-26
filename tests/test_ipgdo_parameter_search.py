from pathlib import Path

import pandas as pd
import pytest

from cpdp.oversampling_validation.parameter_search import (
    ParameterSearchEngine,
    ParameterSet,
    aggregate_source_cv_project_metrics,
    interaction_candidates,
    load_parameter_search_config,
    rank_candidates,
    stage_one_candidates,
    stage_three_candidates,
    stage_two_candidates,
    unique_parameter_sets,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEARCH_CONFIG = PROJECT_ROOT / "configs" / "ipgdo_parameter_search.yaml"
METRICS = ("auc", "auprc", "mcc", "f1", "gmean", "popt20")


def _project_rows(trial_id, first, second):
    rows = []
    for family, target, values in (
        ("NASA", "CM1", first),
        ("AEEEM", "EQ", second),
    ):
        rows.append(
            {
                "trial_id": trial_id,
                "family": family,
                "target": target,
                **dict(zip(METRICS, values)),
            }
        )
    return rows


def test_search_schedule_uses_thirteen_sequential_unique_configurations() -> None:
    config = load_parameter_search_config(SEARCH_CONFIG)
    assert config.initial == ParameterSet(0.6, 0.4, 0.1)
    stage_one = stage_one_candidates(config)
    stage_two = stage_two_candidates(config, best_target_ratio=0.25)
    stage_three = stage_three_candidates(
        config,
        best_target_ratio=0.25,
        best_physics_timestep_fraction=0.4,
    )

    unique = unique_parameter_sets((*stage_one, *stage_two, *stage_three))

    assert len(stage_one) == 5
    assert len(stage_two) == 5
    assert len(stage_three) == 5
    assert len(unique) == 13


def test_interaction_check_is_two_by_two_by_two_and_reuses_trials() -> None:
    config = load_parameter_search_config(SEARCH_CONFIG)
    sequential = unique_parameter_sets(
        (
            *stage_one_candidates(config),
            *stage_two_candidates(config, best_target_ratio=0.25),
            *stage_three_candidates(
                config,
                best_target_ratio=0.25,
                best_physics_timestep_fraction=0.4,
            ),
        )
    )
    interactions = interaction_candidates(
        target_ratios=(0.15, 0.25),
        physics_timestep_fractions=(0.4, 0.5),
        condition_dropouts=(0.05, 0.1),
    )
    all_trials = unique_parameter_sets((*sequential, *interactions))

    assert len(interactions) == 8
    assert len(all_trials) <= 20
    assert len(all_trials) < len(sequential) + len(interactions)


def test_rank_candidates_uses_six_metric_project_level_mean_rank() -> None:
    stronger = ParameterSet(0.25, 0.4, 0.1)
    weaker = ParameterSet(0.8, 0.8, 0.3)
    frame = pd.DataFrame(
        _project_rows(stronger.trial_id, (0.82,) * 6, (0.80,) * 6)
        + _project_rows(weaker.trial_id, (0.78,) * 6, (0.79,) * 6)
    )
    baseline = pd.DataFrame(
        [
            {"family": "NASA", "target": "CM1", **dict.fromkeys(METRICS, 0.80)},
            {"family": "AEEEM", "target": "EQ", **dict.fromkeys(METRICS, 0.79)},
        ]
    )

    ranking = rank_candidates(
        frame,
        baseline,
        {stronger.trial_id: stronger, weaker.trial_id: weaker},
        METRICS,
        {"auc": 0.05, "auprc": 0.05, "popt20": 0.05},
    )

    assert ranking.iloc[0]["trial_id"] == stronger.trial_id
    assert ranking.iloc[0]["mean_rank"] == 1.0
    assert ranking.iloc[0]["ranked_cells"] == 12


def test_safety_constraints_override_an_unsafe_first_rank() -> None:
    unsafe = ParameterSet(0.8, 0.8, 0.3)
    safe = ParameterSet(0.25, 0.5, 0.1)
    unsafe_values = (0.79, 0.79, 0.95, 0.95, 0.95, 0.95)
    safe_values = (0.799, 0.799, 0.80, 0.80, 0.80, 0.799)
    frame = pd.DataFrame(
        _project_rows(unsafe.trial_id, unsafe_values, unsafe_values)
        + _project_rows(safe.trial_id, safe_values, safe_values)
    )
    baseline = pd.DataFrame(
        [
            {"family": "NASA", "target": "CM1", **dict.fromkeys(METRICS, 0.80)},
            {"family": "AEEEM", "target": "EQ", **dict.fromkeys(METRICS, 0.80)},
        ]
    )

    ranking = rank_candidates(
        frame,
        baseline,
        {unsafe.trial_id: unsafe, safe.trial_id: safe},
        METRICS,
        {"auc": 0.005, "auprc": 0.005, "popt20": 0.01},
    )

    assert ranking.iloc[0]["trial_id"] == safe.trial_id
    unsafe_row = ranking[ranking["trial_id"] == unsafe.trial_id].iloc[0]
    assert bool(unsafe_row["safety_eligible"]) is False
    assert set(unsafe_row["safety_violations"].split(",")) == {"auc", "auprc"}


def test_source_cv_project_metrics_average_folds_and_seeds() -> None:
    rows = []
    for family, project in (("NASA", "CM1"), ("AEEEM", "EQ")):
        for seed in (11, 12):
            for fold in (0, 1):
                value = 0.7 + 0.01 * (seed - 11) + 0.02 * fold
                rows.append(
                    {
                        "protocol": "source_cv",
                        "variant": "full",
                        "family": family,
                        "evaluated_project": f"{family}/{project}",
                        "seed": seed,
                        "fold": fold,
                        **{
                            f"augmented_{metric}": value
                            for metric in METRICS
                        },
                    }
                )

    aggregated = aggregate_source_cv_project_metrics(
        pd.DataFrame(rows),
        trial_id="trial",
        variant="full",
        seeds=(11, 12),
        folds=(0, 1),
        projects=(("NASA", "CM1"), ("AEEEM", "EQ")),
        metrics=METRICS,
    )

    assert len(aggregated) == 2
    assert set(aggregated["target"]) == {"CM1", "EQ"}
    assert aggregated["auc"].tolist() == pytest.approx([0.715, 0.715])


def test_generated_trial_uses_source_cv_and_only_searches_selected_parameters() -> None:
    config = load_parameter_search_config(SEARCH_CONFIG)
    engine = ParameterSearchEngine(config, workers=1)
    parameters = ParameterSet(0.4, 0.65, 0.2)

    trial = engine._trial_mapping(parameters)
    overrides = trial["variants"]["full"]["overrides"]

    assert overrides["target_minority_ratio"] == 0.4
    assert overrides["candidate_multiplier"] == 3.5
    assert overrides["diffusion"]["physics_warmup_fraction"] == 0.2
    assert overrides["diffusion"]["formula_semantic_weight"] == 0.05
    assert overrides["diffusion"]["defect_anchor_start_fraction"] == 0.5
    assert overrides["diffusion"]["guidance_scale"] == 1.5
    assert overrides["diffusion"]["physics_timestep_fraction"] == 0.65
    assert overrides["diffusion"]["condition_dropout"] == 0.2
    assert overrides["diffusion"]["formula_consistency_weight"] == 0.0
    assert overrides["formula"]["minimum_reliability"] == 0.0
    assert overrides["formula"]["use_reliability_weighting"] is False
    assert trial["profiles"]["search"]["seeds"] == list(config.all_seeds)
    assert trial["profiles"]["search"]["source_cv"]["enabled"] is True
    assert trial["profiles"]["search"]["cpdp"]["enabled"] is False
