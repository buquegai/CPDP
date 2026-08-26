from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cpdp.config import load_config
from cpdp.methods.registry import available_methods
from cpdp.oversampling.config import OversamplingConfig
from cpdp.oversampling.types import AugmentationResult
from cpdp.oversampling_validation.config import (
    CPDPConfig,
    ClassifierConfig,
    QualityConfig,
    SourceCVConfig,
    ValidationProfile,
    load_validation_config,
)
from cpdp.oversampling_validation.launcher import (
    ALL_SEEDS,
    LaunchOptions,
    build_run_command,
    select_profile_seeds,
)
from cpdp.oversampling_validation.protocols import (
    run_cpdp_target,
    run_source_cv_fold,
)
from cpdp.oversampling_validation.quality import evaluate_synthetic_quality
from cpdp.oversampling_validation.store import ValidationStore
from cpdp.oversampling_validation.strategies import (
    create_strategy,
    effective_strategy_config,
)
from cpdp.types import LabeledProject, PredictionResult, ProjectFeatures


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _project(family, name, seed=1, rows=30):
    random = np.random.default_rng(seed)
    labels = np.tile(np.asarray([0, 1], dtype=np.int64), rows // 2)
    values = random.normal(size=(len(labels), 3))
    values[:, 0] += labels * 0.8
    return LabeledProject(
        features=ProjectFeatures(
            family=family,
            name=name,
            values=values,
            feature_names=("loc", "wmc", "cbo"),
            effort=np.abs(values[:, 0]),
            effort_name="loc",
            row_ids=np.arange(len(labels), dtype=np.int64),
        ),
        labels=labels,
    )


def _profile():
    return ValidationProfile(
        name="test",
        oversampling_profile="smoke",
        seeds=(17,),
        variants=("none",),
        source_cv=SourceCVConfig(folds=3, validation_fraction=0.2),
        cpdp=CPDPConfig(validation_fraction=0.2),
        classifier=ClassifierConfig(max_iter=200),
        quality=QualityConfig(max_samples=50, knn_neighbors=3),
    )


def _formal_profile():
    return ValidationProfile(
        name="formal-test",
        oversampling_profile="smoke",
        seeds=(17,),
        variants=("none", "random", "smote", "tabddpm"),
        source_cv=SourceCVConfig(enabled=False, folds=3, validation_fraction=0.2),
        cpdp=CPDPConfig(validation_fraction=0.2),
        classifier=ClassifierConfig(
            kind="formal_kan_jepa_tabr",
            method="ipgdo_kan_jepa",
            predictor={"device": "cpu"},
        ),
        quality=QualityConfig(max_samples=50, knn_neighbors=3),
    )


def test_validation_strategies_do_not_pollute_formal_method_registry() -> None:
    formal = set(available_methods())

    create_strategy("none", {}, {})
    create_strategy("random", {"target_minority_ratio": 1.0}, {})

    assert set(available_methods()) == formal
    assert "none" not in formal
    assert "smote" not in formal


def test_random_strategy_marks_synthetic_rows_and_preserves_real_rows() -> None:
    project = _project("TEST", "source")
    imbalanced = LabeledProject(project.features, np.asarray([0] * 24 + [1] * 6))
    strategy = create_strategy(
        "random", {"target_minority_ratio": 1.0}, {"max_synthetic_per_project": 5}
    )

    result = strategy.fit_resample((imbalanced,), 11)
    augmented = result.projects[0]

    assert augmented.features.n_samples == imbalanced.features.n_samples + 5
    np.testing.assert_array_equal(
        augmented.features.row_ids[: imbalanced.features.n_samples],
        imbalanced.features.row_ids,
    )
    assert np.all(augmented.features.row_ids[-5:] < 0)
    assert np.all(augmented.labels[-5:] == 1)


def test_no_synthetic_quality_is_missing_with_explicit_reason() -> None:
    train = _project("TEST", "source", seed=2)
    heldout = _project("TEST", "source", seed=3)

    result = evaluate_synthetic_quality(
        train, heldout, train, QualityConfig(), seed=5
    )

    assert result.metadata["reason"] == "no_synthetic_samples"
    assert all(value is None for value in result.metrics.values())


class _RecordingPassThrough:
    def __init__(self):
        self.seen_keys = []
        self.seen_seed = None

    def fit_resample(self, sources, seed):
        self.seen_keys = [project.key for project in sources]
        self.seen_seed = seed
        return AugmentationResult(tuple(sources), {"total_synthetic_selected": 0})


def test_source_cv_none_has_zero_paired_gain() -> None:
    projects = tuple(_project("TEST", f"p{index}", seed=index + 1) for index in range(3))
    strategy = _RecordingPassThrough()

    result = run_source_cv_fold(projects, strategy, _profile(), seed=17, fold=0)

    assert strategy.seen_keys == [project.key for project in projects]
    assert all(value == pytest.approx(0.0) for value in result.utility["delta_metrics"].values())
    assert all(item["metadata"]["reason"] == "no_synthetic_samples" for item in result.quality)


def test_source_cv_can_isolate_oversampling_randomness() -> None:
    projects = tuple(_project("TEST", f"p{index}", seed=index + 1) for index in range(3))
    strategy = _RecordingPassThrough()

    run_source_cv_fold(
        projects,
        strategy,
        _profile(),
        seed=999,
        fold=0,
        split_seed=11,
        oversampling_seed=22,
        classifier_seed=33,
        quality_seed=44,
    )

    assert strategy.seen_seed == 22


def test_cpdp_never_passes_target_to_oversampler() -> None:
    sources = tuple(_project("TEST", f"source{index}", seed=index + 5) for index in range(3))
    target = _project("TEST", "target", seed=99)
    strategy = _RecordingPassThrough()

    result = run_cpdp_target(sources, target, strategy, _profile(), seed=17)

    assert target.key not in strategy.seen_keys
    assert set(strategy.seen_keys) == {project.key for project in sources}
    assert all(value == pytest.approx(0.0) for value in result.utility["delta_metrics"].values())


def test_formal_cpdp_runs_one_downstream_prediction(monkeypatch) -> None:
    sources = tuple(_project("TEST", f"source{index}", seed=index + 5) for index in range(3))
    target = _project("TEST", "target", seed=99)
    strategy = _RecordingPassThrough()
    calls = []

    def fake_predict(train_projects, heldout, config, seed, validation_projects=()):
        calls.append((tuple(project.key for project in train_projects), heldout.key, seed))
        return PredictionResult(
            scores=np.linspace(0.1, 0.9, len(heldout.labels)),
            threshold=0.5,
            metadata={"predictor": "fake-kan-jepa-tabr"},
            evaluated_row_ids=heldout.features.row_ids,
        )

    monkeypatch.setattr(
        "cpdp.oversampling_validation.protocols.predict_validation_target",
        fake_predict,
    )
    result = run_cpdp_target(sources, target, strategy, _formal_profile(), seed=17)

    assert len(calls) == 1
    assert calls[0][1:] == (target.key, 17)
    assert "baseline" not in result.utility
    assert result.utility["augmented"]["classifier"]["predictor"] == (
        "fake-kan-jepa-tabr"
    )
    assert result.quality == ()


def test_tabddpm_strategy_generates_without_knn(monkeypatch) -> None:
    class FakeDiffusion:
        def __init__(self, input_dim, project_count, config, device):
            self.input_dim = input_dim

        def fit(self, values, project_ids, seed):
            return {"seed": seed, "training_rows": len(values)}

        def sample_defects(self, count, project_id):
            return np.zeros((count, self.input_dim), dtype=np.float64)

    monkeypatch.setattr(
        "cpdp.oversampling_validation.strategies.ProjectConditionedTabularDDPM",
        FakeDiffusion,
    )
    sources = []
    for index in range(2):
        project = _project("TEST", f"source{index}", seed=index + 3)
        sources.append(
            LabeledProject(project.features, np.asarray([0] * 24 + [1] * 6))
        )
    strategy = create_strategy("tabddpm", {}, {"target_minority_ratio": 1.0})

    result = strategy.fit_resample(tuple(sources), seed=17)

    assert result.metadata["name"] == "TabDDPM"
    assert result.metadata["total_synthetic_selected"] == 36
    assert all(
        item["knn"] == {"enabled": False, "reason": "tabddpm_baseline"}
        for item in result.metadata["projects"]
    )
    assert all(np.sum(project.features.row_ids < 0) == 18 for project in result.projects)


def test_summary_pairs_single_pass_variants_with_none_baseline(tmp_path) -> None:
    store = ValidationStore(tmp_path, "fingerprint", "full")

    def payload(variant, auc):
        metrics = {
            "auc": auc,
            "auprc": auc,
            "mcc": auc,
            "f1": auc,
            "gmean": auc,
            "popt20": auc,
        }
        return {
            "fingerprint": "fingerprint",
            "profile": "full",
            "protocol": "cpdp",
            "variant": variant,
            "family": "TEST",
            "target": "target",
            "seed": 17,
            "fold": None,
            "status": "success",
            "utility": {
                "paired_baseline": False,
                "augmented": {
                    "threshold": 0.5,
                    "metrics": metrics,
                    "per_project_metrics": {"TEST/target": metrics},
                    "classifier": {},
                },
                "augmentation_metadata": {"total_synthetic_selected": 0},
                "duration_seconds": 1.0,
            },
            "quality": [],
        }

    store.save(store.run_path("cpdp", "none", "TEST", 17, target="target"), payload("none", 0.6))
    store.save(store.run_path("cpdp", "smote", "TEST", 17, target="target"), payload("smote", 0.7))

    paths = store.summarize()
    utility = pd.read_csv(paths["utility_runs"])
    smote = utility.loc[utility["variant"] == "smote"].iloc[0]
    assert smote["baseline_auc"] == pytest.approx(0.6)
    assert smote["augmented_auc"] == pytest.approx(0.7)
    assert smote["delta_auc"] == pytest.approx(0.1)


def test_validation_launcher_builds_independent_command(tmp_path) -> None:
    options = LaunchOptions(
        profile="smoke",
        protocols=("source_cv",),
        variants=("full",),
        family="NASA",
        target=None,
        seeds=(2026072201, 2026072202),
        max_tasks=1,
        force=True,
        fail_fast=True,
    )

    command = build_run_command("python", tmp_path, options)

    assert command[1] == str(tmp_path / "run_oversampling_validation.py")
    assert "run_experiment.py" not in command
    assert command.count("--seed") == 2
    assert command[-2:] == ["--force", "--fail-fast"]


def test_validation_launcher_selects_requested_seed_prefix() -> None:
    configured = (2026072201, 2026072202, 2026072203, 2026072204)

    assert select_profile_seeds(configured, ALL_SEEDS, 2) == configured[:2]
    assert select_profile_seeds(configured, "2026072203", 4) == (2026072203,)


def test_validation_launcher_rejects_invalid_seed_count() -> None:
    with pytest.raises(ValueError, match="运行种子数量"):
        select_profile_seeds((2026072201, 2026072202), ALL_SEEDS, 3)


def test_final_ipgdo_parameters_are_frozen_without_changing_baselines() -> None:
    defaults = OversamplingConfig.from_mapping({})
    assert defaults.target_minority_ratio == pytest.approx(1.0)
    assert defaults.diffusion.hidden_dim == 256
    assert defaults.diffusion.residual_blocks == 4
    assert defaults.diffusion.timesteps == 200
    assert defaults.diffusion.epochs == 300
    assert defaults.diffusion.learning_rate == pytest.approx(0.001)
    assert defaults.diffusion.physics_warmup_fraction == pytest.approx(0.2)
    assert defaults.diffusion.formula_consistency_weight == pytest.approx(0.0)
    assert defaults.diffusion.formula_semantic_weight == pytest.approx(0.05)
    assert defaults.formula.max_terms == 10
    assert defaults.formula.use_reliability_weighting is False

    experiment = load_config(PROJECT_ROOT / "configs" / "experiment.yaml")
    method_params = experiment.method_params("full", "ipgdo_kan_jepa")
    base = method_params["oversampling"]

    assert method_params["result_signature"] == (
        "ipgdo-kan-jepa-pinn-ipgdo-r1p0-v3"
    )
    assert experiment.repetitions == 10
    assert len(experiment.seeds) == 10
    assert method_params["oversampling_reuse"] == {"enabled": False}
    assert base["target_minority_ratio"] == pytest.approx(1.0)
    assert base["diffusion"]["hidden_dim"] == 256
    assert base["diffusion"]["residual_blocks"] == 4
    assert base["diffusion"]["timesteps"] == 200
    assert base["diffusion"]["epochs"] == 300
    assert base["diffusion"]["learning_rate"] == pytest.approx(0.001)
    assert base["diffusion"]["formula_consistency_weight"] == pytest.approx(0.0)
    assert base["diffusion"]["formula_semantic_weight"] == pytest.approx(0.05)
    assert base["formula"]["max_terms"] == 10
    assert base["formula"]["use_reliability_weighting"] is False
    assert "knn" not in base

    transformer = experiment.method_params("full", "ipgdo_transformer_jepa")
    assert transformer["oversampling"] == base
    assert transformer["result_signature"] == (
        "ipgdo-transformer-jepa-pinn-ipgdo-r1p0-v3"
    )

    for method_name in ("ipgdo_kan_jepa", "ipgdo_transformer_jepa"):
        smoke = experiment.method_params("smoke", method_name)["oversampling"]
        assert smoke["target_minority_ratio"] == pytest.approx(1.0)
        assert smoke["formula"]["max_terms"] == 10
        assert smoke["diffusion"]["formula_semantic_weight"] == pytest.approx(
            0.05
        )
        assert "knn" not in smoke

    validation = load_validation_config(
        PROJECT_ROOT / "configs" / "oversampling_validation.yaml"
    )
    validation_profile = validation.profile("full")
    assert validation_profile.variants == ("none", "random", "smote", "tabddpm")
    assert not validation_profile.source_cv.enabled
    assert validation_profile.cpdp.enabled
    assert validation_profile.classifier.kind == "formal_kan_jepa_tabr"
    assert validation_profile.classifier.method == "ipgdo_kan_jepa"
    assert validation_profile.classifier.predictor == method_params["predictor"]
    full_variant = validation.variant("full")
    full_effective = effective_strategy_config(
        full_variant.kind, base, full_variant.overrides
    )["config"]
    assert full_effective["target_minority_ratio"] == pytest.approx(1.0)
    assert full_effective["diffusion"]["hidden_dim"] == 256
    assert full_effective["diffusion"]["residual_blocks"] == 4
    assert full_effective["diffusion"]["timesteps"] == 200
    assert full_effective["diffusion"]["epochs"] == 300
    assert full_effective["diffusion"]["learning_rate"] == pytest.approx(0.001)
    assert full_effective["formula"]["max_terms"] == 10
    assert "knn" not in full_effective

    tabddpm_variant = validation.variant("tabddpm")
    tabddpm_effective = effective_strategy_config(
        tabddpm_variant.kind, base, tabddpm_variant.overrides
    )["config"]
    assert tabddpm_effective == full_effective

    for variant_name in ("random", "smote", "tabddpm"):
        variant = validation.variant(variant_name)
        effective = effective_strategy_config(
            variant.kind, base, variant.overrides
        )["config"]
        assert effective["target_minority_ratio"] == pytest.approx(1.0)
