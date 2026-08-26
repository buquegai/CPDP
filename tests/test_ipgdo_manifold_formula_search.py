from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
import yaml

import cpdp.oversampling_validation.manifold_formula_search as search_module
from cpdp.oversampling_validation.manifold_formula_plot import (
    create_manifold_formula_figure,
)
from cpdp.oversampling_validation.manifold_formula_search import (
    ManifoldFormulaParameterSet,
    ManifoldFormulaSearchEngine,
    ManifoldInterval,
    aggregate_cpdp_target_metrics,
    interaction_candidates,
    load_manifold_formula_search_config,
    stage_one_candidates,
    stage_three_candidates,
    stage_two_candidates,
)
from cpdp.oversampling_validation.parameter_search import unique_parameter_sets


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEARCH_CONFIG = PROJECT_ROOT / "configs" / "ipgdo_manifold_formula_search.yaml"
METRICS = ("auc", "auprc", "mcc", "f1", "gmean", "popt20")


def test_fourth_round_uses_current_best_baseline_and_expected_schedule() -> None:
    config = load_manifold_formula_search_config(SEARCH_CONFIG)

    assert config.initial == ManifoldFormulaParameterSet(0.025, 0.975, 5, 10)
    assert config.fixed_overrides["target_minority_ratio"] == pytest.approx(0.6)
    assert config.fixed_overrides["candidate_multiplier"] == pytest.approx(3.5)
    diffusion = config.fixed_overrides["diffusion"]
    assert diffusion["physics_warmup_fraction"] == pytest.approx(0.2)
    assert diffusion["formula_semantic_weight"] == pytest.approx(0.05)
    assert diffusion["defect_anchor_start_fraction"] == pytest.approx(0.5)
    assert diffusion["guidance_scale"] == pytest.approx(1.5)
    assert diffusion["physics_timestep_fraction"] == pytest.approx(0.4)
    assert diffusion["condition_dropout"] == pytest.approx(0.1)
    assert len(config.final_cpdp.seeds) == 10

    first = stage_one_candidates(config)
    second = stage_two_candidates(config, ManifoldInterval(0.05, 0.95))
    third = stage_three_candidates(config, ManifoldInterval(0.05, 0.95), 5)
    assert len(first) == len(second) == len(third) == 5
    assert len(unique_parameter_sets((*first, *second, *third))) == 13

    interactions = interaction_candidates(
        (ManifoldInterval(0.05, 0.95), ManifoldInterval(0.1, 0.9)),
        (3, 5),
        (8, 10),
    )
    assert len(interactions) == 8


def test_generated_search_and_final_cpdp_configs_use_only_selected_parameters() -> None:
    config = load_manifold_formula_search_config(SEARCH_CONFIG)
    engine = ManifoldFormulaSearchEngine(config, workers=1)
    parameters = ManifoldFormulaParameterSet(0.1, 0.9, 7, 8)

    trial = engine._trial_mapping(parameters)
    overrides = trial["variants"]["full"]["overrides"]
    assert overrides["target_minority_ratio"] == pytest.approx(0.6)
    assert overrides["candidate_multiplier"] == pytest.approx(3.5)
    assert overrides["manifold"] == {
        "lower_quantile": 0.1,
        "upper_quantile": 0.9,
        "neighbors": 7,
    }
    assert overrides["formula"]["max_terms"] == 8
    assert trial["profiles"]["search"]["source_cv"]["enabled"] is True
    assert trial["profiles"]["search"]["cpdp"]["enabled"] is False

    final = engine._final_cpdp_mapping(parameters)
    profile = final["profiles"]["final_cpdp"]
    assert profile["seeds"] == list(config.all_seeds)
    assert profile["source_cv"]["enabled"] is False
    assert profile["cpdp"]["enabled"] is True
    assert final["variants"]["full"]["overrides"] == overrides


def test_cpdp_target_aggregation_requires_and_returns_all_seed_results() -> None:
    rows = []
    targets = (("NASA", "CM1"), ("AEEEM", "EQ"))
    seeds = (11, 12)
    for target_index, (family, target) in enumerate(targets):
        for seed_index, seed in enumerate(seeds):
            value = 0.7 + target_index * 0.05 + seed_index * 0.02
            rows.append(
                {
                    "protocol": "cpdp",
                    "variant": "full",
                    "family": family,
                    "target": target,
                    "seed": seed,
                    **{f"augmented_{metric}": value for metric in METRICS},
                }
            )

    runs, means, macro = aggregate_cpdp_target_metrics(
        pd.DataFrame(rows),
        variant="full",
        seeds=seeds,
        targets=targets,
        metrics=METRICS,
    )

    assert len(runs) == 4
    assert len(means) == 2
    assert means["seed_count"].tolist() == [2, 2]
    assert means["auc"].tolist() == pytest.approx([0.71, 0.76])
    assert macro["macro_auc"] == pytest.approx(0.735)


def test_search_run_automatically_invokes_final_cpdp(monkeypatch, tmp_path) -> None:
    original = load_manifold_formula_search_config(SEARCH_CONFIG)
    final_cpdp = replace(original.final_cpdp, output_dir=tmp_path / "cpdp")
    config = replace(
        original,
        output_dir=tmp_path / "search",
        validation_output_dir=tmp_path / "validation",
        final_cpdp=final_cpdp,
    )
    engine = ManifoldFormulaSearchEngine(config, workers=1)
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
        "create_manifold_formula_figure",
        lambda output_dir, winner: (tmp_path / "figure.png", tmp_path / "figure.pdf"),
    )

    winner = engine.run()

    assert events[-1] == ("final_cpdp", winner.trial_id, config.all_seeds)
    assert events[-2][0] == "stage_05_winner_confirmation"
    payload = yaml.safe_load((config.output_dir / "best_params.yaml").read_text("utf-8"))
    assert payload["final_cpdp"]["successful_tasks"] == 250


def test_fourth_round_parameter_plot_is_created(tmp_path) -> None:
    pd.DataFrame(
        {
            "manifold_lower_quantile": [0.01, 0.05, 0.1],
            "manifold_upper_quantile": [0.99, 0.95, 0.9],
            "mean_rank": [2.1, 1.4, 1.8],
            "selection_position": [3, 1, 2],
        }
    ).to_csv(tmp_path / "stage_01_manifold_interval_ranking.csv", index=False)
    pd.DataFrame(
        {
            "manifold_neighbors": [3, 5, 7],
            "mean_rank": [1.9, 1.3, 1.7],
            "selection_position": [3, 1, 2],
        }
    ).to_csv(tmp_path / "stage_02_manifold_neighbors_ranking.csv", index=False)
    pd.DataFrame(
        {
            "formula_max_terms": [6, 8, 10],
            "mean_rank": [2.0, 1.5, 1.6],
            "selection_position": [3, 1, 2],
        }
    ).to_csv(tmp_path / "stage_03_formula_max_terms_ranking.csv", index=False)

    png_path, pdf_path = create_manifold_formula_figure(
        tmp_path,
        {
            "manifold_lower_quantile": 0.05,
            "manifold_upper_quantile": 0.95,
            "manifold_neighbors": 5,
            "formula_max_terms": 8,
        },
        dpi=120,
    )

    assert png_path.is_file()
    assert pdf_path.is_file()
