from inspect import signature

import numpy as np
import pytest

from cpdp.methods.proposed_pipeline import run_oversampler_then_predictor
from cpdp.oversampling.config import KNNFilterConfig
from cpdp.oversampling.formula_miner import FrozenFormulaGuidance, SparseFormulaMiner
from cpdp.oversampling.legacy_config import FormulaConfig, RoleConfig
from cpdp.oversampling.manifold_filter import DefectManifoldFilter
from cpdp.oversampling.pipeline import IPGDOOversampler
from cpdp.oversampling.semantic_roles import SemanticRoleResolver
from cpdp.oversampling.symbolic_library import FittedSymbolicLibrary
from cpdp.oversampling.types import AugmentationResult, FormulaBank
from cpdp.types import (
    LabeledProject,
    PredictionResult,
    ProjectFeatures,
    UnlabeledProject,
)


def _project(family, name, feature_names, values, labels):
    array = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    effort_index = feature_names.index(feature_names[0])
    features = ProjectFeatures(
        family=family,
        name=name,
        values=array,
        feature_names=tuple(feature_names),
        effort=array[:, effort_index],
        effort_name=feature_names[0],
        row_ids=np.arange(len(array)),
    )
    return LabeledProject(features, labels)


def test_aeeem_roles_use_sparse_semantic_aggregation() -> None:
    names = (
        "ck_oo_numberOfLinesOfCode",
        "LDHH_numberOfLinesOfCode",
        "WCHU_numberOfLinesOfCode",
        "ck_oo_wmc",
        "LDHH_wmc",
        "WCHU_wmc",
        "ck_oo_cbo",
        "LDHH_fanOut",
        "WCHU_rfc",
    )
    random = np.random.default_rng(7)
    values = random.lognormal(size=(80, len(names)))
    labels = (values[:, 0] + values[:, 3] + values[:, 6] > 4.0).astype(np.int64)

    mapping = SemanticRoleResolver(RoleConfig(max_components=2)).fit(
        values, labels, names
    )

    assert mapping.names == ("S", "C", "K")
    assert all(len(role.components) <= 2 for role in mapping.roles)
    assert all(
        sum(component.weight for component in role.components) == pytest.approx(1.0)
        for role in mapping.roles
    )


def test_relink_library_omits_unavailable_coupling_terms() -> None:
    names = ("CountLineCode", "SumCyclomatic", "AvgCyclomatic", "SumEssential")
    random = np.random.default_rng(11)
    values = random.lognormal(size=(60, len(names)))
    labels = (values[:, 0] + values[:, 1] > 3.0).astype(np.int64)
    mapping = SemanticRoleResolver(RoleConfig()).fit(values, labels, names)
    library = FittedSymbolicLibrary.fit(values, mapping)

    assert mapping.names == ("S", "C", "E")
    assert all("K" not in term for term in library.term_names)
    assert any("safe_log_ratio" in term for term in library.term_names)


def test_formula_guidance_is_frozen_and_differentiable() -> None:
    torch = pytest.importorskip("torch")
    names = ("loc", "wmc", "cbo")
    random = np.random.default_rng(19)
    projects = []
    for index in range(3):
        values = random.lognormal(mean=0.2 * index, size=(50, 3))
        labels = (values[:, 0] + values[:, 1] + 0.5 * values[:, 2] > 3.6).astype(
            np.int64
        )
        projects.append(_project("PROMISE", f"p{index}", names, values, labels))
    all_values = np.vstack([project.features.values for project in projects])
    all_labels = np.concatenate([project.labels for project in projects])
    mapping = SemanticRoleResolver(RoleConfig()).fit(all_values, all_labels, names)
    library = FittedSymbolicLibrary.fit(all_values, mapping)
    bank = SparseFormulaMiner(
        FormulaConfig(minimum_reliability=0.0, max_terms=5)
    ).fit(projects, library)
    guidance = FrozenFormulaGuidance(library, bank)

    predicted = torch.tensor(all_values[:12], dtype=torch.float32, requires_grad=True)
    clean = torch.tensor(all_values[:12], dtype=torch.float32)
    labels = torch.tensor(all_labels[:12], dtype=torch.long)
    project_ids = torch.zeros(12, dtype=torch.long)
    valid = torch.ones(12, dtype=torch.bool)
    consistency, semantic = guidance.losses(
        predicted, clean, labels, project_ids, valid
    )
    (consistency + semantic).backward()

    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()
    assert bank.global_coefficients.flags.writeable is False
    assert bank.project_coefficients.flags.writeable is False


def test_formula_guidance_can_disable_reliability_weighting() -> None:
    torch = pytest.importorskip("torch")

    class _IdentityLibrary:
        @staticmethod
        def transform_torch(values):
            return values[:, :1]

    bank = FormulaBank(
        term_names=("x",),
        term_centers=np.zeros(1),
        term_scales=np.ones(1),
        global_coefficients=np.ones(1),
        global_intercept=0.0,
        project_coefficients=np.ones((1, 1)),
        project_intercepts=np.zeros(1),
        project_names=("TEST/source",),
        reliability=0.25,
        validation_auc=0.625,
        enabled=True,
    )
    predicted = torch.tensor([[0.2], [0.8]], dtype=torch.float32, requires_grad=True)
    clean = torch.tensor([[0.1], [0.9]], dtype=torch.float32)
    labels = torch.tensor([0, 1], dtype=torch.long)
    project_ids = torch.zeros(2, dtype=torch.long)
    valid = torch.ones(2, dtype=torch.bool)

    weighted = FrozenFormulaGuidance(
        _IdentityLibrary(), bank, use_reliability_weighting=True
    ).losses(predicted, clean, labels, project_ids, valid)
    unweighted = FrozenFormulaGuidance(
        _IdentityLibrary(), bank, use_reliability_weighting=False
    ).losses(predicted, clean, labels, project_ids, valid)

    torch.testing.assert_close(weighted[0], unweighted[0] * bank.reliability)
    torch.testing.assert_close(weighted[1], unweighted[1] * bank.reliability)
    sum(unweighted).backward()
    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()


def test_knn_filter_rejects_too_close_and_too_far_candidates() -> None:
    random = np.random.default_rng(7)
    real_defects = random.normal(size=(80, 3))
    ordinary_candidates = random.normal(size=(20, 3))
    candidates = np.vstack(
        (real_defects[0], ordinary_candidates[1], np.full(3, 20.0))
    )
    manifold = DefectManifoldFilter(real_defects, KNNFilterConfig())

    evaluation = manifold.evaluate(candidates)

    assert evaluation.accepted.tolist() == [False, True, False]


def test_oversampler_api_cannot_receive_target_data() -> None:
    parameters = signature(
        IPGDOOversampler.fit_resample
    ).parameters

    assert tuple(parameters) == ("self", "sources", "seed")


def test_predictor_can_be_replaced_without_oversampler_changes() -> None:
    source = _project(
        "PROMISE",
        "source",
        ("loc", "wmc", "cbo"),
        ((1.0, 1.0, 1.0), (2.0, 2.0, 2.0)),
        (0, 1),
    )
    target_project = _project(
        "PROMISE",
        "target",
        ("loc", "wmc", "cbo"),
        ((1.5, 1.5, 1.5),),
        (0,),
    )

    class PassThroughOversampler:
        seen_seed = None

        def fit_resample(self, sources, seed):
            self.seen_seed = seed
            return AugmentationResult(tuple(sources), {"seed": seed})

    class StubPredictor:
        def fit_predict(self, sources, target, seed):
            return PredictionResult(np.full(target.features.n_samples, 0.75))

    oversampler = PassThroughOversampler()
    result = run_oversampler_then_predictor(
        oversampler,
        StubPredictor(),
        (source,),
        target_project.without_labels(),
        123,
        oversampling_seed=456,
    )

    np.testing.assert_array_equal(result.scores, np.asarray([0.75]))
    assert oversampler.seen_seed == 456
    assert result.metadata["oversampling"] == {"seed": 456}
    assert result.metadata["random_seeds"] == {
        "oversampling_seed": 456,
        "predictor_seed": 123,
    }
