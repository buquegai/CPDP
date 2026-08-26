from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
import yaml

import cpdp.oversampling_validation.formula_optimization_search as search_module
from cpdp.oversampling_validation.formula_optimization_plot import (
    create_formula_optimization_figure,
)
from cpdp.oversampling_validation.formula_optimization_search import (
    FormulaOptimizationParameterSet,
    FormulaOptimizationSearchEngine,
    interaction_candidates,
    load_formula_optimization_search_config,
    stage_one_candidates,
    stage_three_candidates,
    stage_two_candidates,
)
from cpdp.oversampling_validation.parameter_search import unique_parameter_sets


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEARCH_CONFIG = PROJECT_ROOT / "configs" / "ipgdo_formula_optimization_search.yaml"
METRICS = ("auc", "auprc", "mcc", "f1", "gmean", "popt20")


def test_fifth_round_uses_current_best_baseline_and_expected_schedule() -> None:
    config = load_formula_optimization_search_config(SEARCH_CONFIG)

    assert config.initial == FormulaOptimizationParameterSet(0.1, 0.0005, 300)
    assert config.fixed_overrides["target_minority_ratio"] == pytest.approx(0.6)
    assert config.fixed_overrides["candidate_multiplier"] == pytest.approx(3.5)
    assert config.fixed_overrides["formula"]["max_terms"] == 10
    diffusion = config.fixed_overrides["diffusion"]
    assert diffusion["physics_warmup_fraction"] == pytest.approx(0.2)
    assert diffusion["formula_semantic_weight"] == pytest.approx(0.05)
    assert diffusion["defect_anchor_start_fraction"] == pytest.approx(0.5)
    assert diffusion["guidance_scale"] == pytest.approx(1.5)
    assert diffusion["physics_timestep_fraction"] == pytest.approx(0.4)
    assert diffusion["condition_dropout"] == pytest.approx(0.1)
    assert config.fixed_overrides["manifold"] == {
        "lower_quantile": 0.025,
        "upper_quantile": 0.975,
        "neighbors": 5,
    }
    assert len(config.final_cpdp.seeds) == 10

    first = stage_one_candidates(config)
    second = stage_two_candidates(config, 0.05)
    third = stage_three_candidates(config, 0.05, 0.00025)
    assert len(first) == len(second) == len(third) == 5
    assert len(unique_parameter_sets((*first, *second, *third))) == 13

    interactions = interaction_candidates(
        (0.05, 0.1),
        (0.00025, 0.0005),
        (225, 300),
    )
    assert len(interactions) == 8


def test_generated_configs_change_only_the_three_selected_parameters() -> None:
    config = load_formula_optimization_search_config(SEARCH_CONFIG)
    engine = FormulaOptimizationSearchEngine(config, workers=1)
    parameters = FormulaOptimizationParameterSet(0.2, 0.00075, 400)

    trial = engine._trial_mapping(parameters)
    overrides = trial["variants"]["full"]["overrides"]
    assert overrides["target_minority_ratio"] == pytest.approx(0.6)
    assert overrides["candidate_multiplier"] == pytest.approx(3.5)
    assert overrides["formula"] == {
        "max_terms": 10,
        "minimum_reliability": 0.0,
        "use_reliability_weighting": False,
        "l1_c": 0.2,
    }
    assert overrides["diffusion"]["learning_rate"] == pytest.approx(0.00075)
    assert overrides["diffusion"]["epochs"] == 400
    assert overrides["manifold"] == config.fixed_overrides["manifold"]
    assert trial["profiles"]["search"]["source_cv"]["enabled"] is True
    assert trial["profiles"]["search"]["cpdp"]["enabled"] is False

    final = engine._final_cpdp_mapping(parameters)
    profile = final["profiles"]["final_cpdp"]
    assert profile["seeds"] == list(config.all_seeds)
    assert profile["source_cv"]["enabled"] is False
    assert profile["cpdp"]["enabled"] is True
    assert final["variants"]["full"]["overrides"] == overrides


def test_search_run_automatically_invokes_final_cpdp(monkeypatch, tmp_path) -> None:
    original = load_formula_optimization_search_config(SEARCH_CONFIG)
    final_cpdp = replace(original.final_cpdp, output_dir=tmp_path / "cpdp")
    config = replace(
        original,
        output_dir=tmp_path / "search",
        validation_output_dir=tmp_path / "validation",
        final_cpdp=final_cpdp,
    )
    engine = FormulaOptimizationSearchEngine(config, workers=1)
    events = []

    def fake_stage(stage_name, candidates, baseline):
        unique = unique_parameter_sets(candidates)
        rows = []
        for position, parameters in enumerate(unique, start=1):
            rows.append(
                {
                    "selection_position": position,
                    "trial_id": parameters.trial_id,
                    **parameters.to_dict(),
                    "mean_rank": float(position),
                    "safety_eligible": True,
                }
            )
        return pd.DataFrame(rows), baseline if baseline is not None else pd.DataFrame()

    def fake_execute(parameters, seeds, stage):
        events.append((stage, parameters.trial_id, tuple(seeds)))
        return None

    def fake_source_metrics(parameters, seeds):
        return pd.DataFrame(
            [
                {
                    "trial_id": parameters.trial_id,
                    "family": "NASA",
                    "target": "CM1",
                    **dict.fromkeys(METRICS, 0.75),
                }
            ]
        )

    def fake_final(parameters):
        events.append(("final_cpdp", parameters.trial_id, config.all_seeds))
        return {
            "config": str(tmp_path / "final.yaml"),
            "fingerprint": "test",
            "output_root": str(tmp_path / "cpdp"),
            "successful_tasks": 250,
            "seed_count": 10,
            "target_count": 25,
            "run_metrics": str(tmp_path / "runs.csv"),
            "target_means": str(tmp_path / "means.csv"),
            "macro_metrics": dict.fromkeys((f"macro_{m}" for m in METRICS), 0.75),
            "validation_summary": {},
        }

    monkeypatch.setattr(engine, "_run_stage", fake_stage)
    monkeypatch.setattr(engine, "execute_trial", fake_execute)
    monkeypatch.setattr(engine, "_source_cv_project_metrics", fake_source_metrics)
    monkeypatch.setattr(engine, "_run_final_cpdp", fake_final)
    monkeypatch.setattr(
        search_module,
        "create_formula_optimization_figure",
        lambda output_dir, winner: (tmp_path / "figure.png", tmp_path / "figure.pdf"),
    )

    winner = engine.run()

    assert events[-1] == ("final_cpdp", winner.trial_id, config.all_seeds)
    assert events[-2][0] == "stage_05_winner_confirmation"
    payload = yaml.safe_load((config.output_dir / "best_params.yaml").read_text("utf-8"))
    assert payload["final_cpdp"]["successful_tasks"] == 250


def test_fifth_round_parameter_plot_is_created(tmp_path) -> None:
    pd.DataFrame(
        {
            "formula_l1_c": [0.05, 0.1, 0.2],
            "mean_rank": [2.1, 1.4, 1.8],
            "selection_position": [3, 1, 2],
        }
    ).to_csv(tmp_path / "stage_01_formula_l1_c_ranking.csv", index=False)
    pd.DataFrame(
        {
            "diffusion_learning_rate": [0.00025, 0.0005, 0.00075],
            "mean_rank": [1.9, 1.3, 1.7],
            "selection_position": [3, 1, 2],
        }
    ).to_csv(tmp_path / "stage_02_diffusion_learning_rate_ranking.csv", index=False)
    pd.DataFrame(
        {
            "diffusion_epochs": [225, 300, 400],
            "mean_rank": [2.0, 1.5, 1.6],
            "selection_position": [3, 1, 2],
        }
    ).to_csv(tmp_path / "stage_03_diffusion_epochs_ranking.csv", index=False)

    png_path, pdf_path = create_formula_optimization_figure(
        tmp_path,
        {
            "formula_l1_c": 0.1,
            "diffusion_learning_rate": 0.0005,
            "diffusion_epochs": 300,
        },
        dpi=120,
    )

    assert png_path.is_file()
    assert pdf_path.is_file()
