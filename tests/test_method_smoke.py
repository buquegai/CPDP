from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from cpdp.config import load_config
from cpdp.data import load_all_projects, sources_for_target
from cpdp.methods.registry import METHOD_NAMES, create_method
from cpdp.randomness import seed_everything
from cpdp.runner import ExperimentRunner


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def experiment_data():
    config = load_config(PROJECT_ROOT / "configs" / "experiment.yaml")
    projects = load_all_projects(config.data)
    target = projects["ReLink/Apache"]
    sources = tuple(sources_for_target(projects, "ReLink", "Apache"))
    return config, sources, target


@pytest.mark.integration
@pytest.mark.parametrize("method_name", METHOD_NAMES)
def test_real_method_smoke(method_name, experiment_data) -> None:
    config, sources, target = experiment_data
    seed = config.seeds[0]
    seed_everything(seed)
    method = create_method(
        method_name,
        PROJECT_ROOT,
        config.method_params("smoke", method_name),
    )

    result = method.fit_predict(sources, target.without_labels(), seed)

    scores = np.asarray(result.scores).reshape(-1)
    assert len(scores) == target.features.n_samples
    assert np.isfinite(scores).all()
    assert np.isfinite(result.threshold)
    if method_name in {
        "ipgdo_kan_jepa",
        "ipgdo_transformer_jepa",
    }:
        assert result.metadata["oversampling"]["total_synthetic_selected"] > 0


@pytest.mark.integration
def test_fedcpdp_parallel_ensemble_matches_serial(experiment_data) -> None:
    config, sources, target = experiment_data
    seed = config.seeds[0]
    base_params = {
        "rounds": 1,
        "local_epochs": 1,
        "ensemble_size": 2,
        "hash_dim": 64,
    }

    seed_everything(seed)
    serial = create_method(
        "fedcpdp", PROJECT_ROOT, {**base_params, "parallel_workers": 1}
    ).fit_predict(sources, target.without_labels(), seed)
    seed_everything(seed)
    parallel = create_method(
        "fedcpdp", PROJECT_ROOT, {**base_params, "parallel_workers": 2}
    ).fit_predict(sources, target.without_labels(), seed)

    np.testing.assert_allclose(serial.scores, parallel.scores, rtol=1e-12, atol=1e-12)
    assert serial.threshold == pytest.approx(parallel.threshold)


@pytest.mark.integration
def test_fedcpdp_outer_seed_batch_matches_individual_runs(experiment_data) -> None:
    config, sources, target = experiment_data
    seeds = config.seeds[:2]
    params = {
        "backend": "torch",
        "device": "cpu",
        "dtype": "float64",
        "seed_strategy": "independent_blake2",
        "rounds": 1,
        "local_epochs": 1,
        "ensemble_size": 2,
        "hash_dim": 64,
        "batch_ensemble_on_device": True,
    }

    batched_method = create_method("fedcpdp", PROJECT_ROOT, params)
    batched = batched_method.fit_predict_many(
        sources, target.without_labels(), seeds
    )
    individual = {
        seed: create_method("fedcpdp", PROJECT_ROOT, params).fit_predict(
            sources, target.without_labels(), seed
        )
        for seed in seeds
    }

    for seed in seeds:
        np.testing.assert_allclose(
            batched[seed].scores,
            individual[seed].scores,
            rtol=1e-11,
            atol=1e-11,
        )
        assert batched[seed].threshold == pytest.approx(
            individual[seed].threshold
        )
        assert (
            batched[seed].metadata["ensemble_seeds"]
            == individual[seed].metadata["ensemble_seeds"]
        )


@pytest.mark.integration
def test_runner_persists_each_fedcpdp_batched_seed(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Runner seed batching is enabled only for CUDA execution")
    base_config = load_config(PROJECT_ROOT / "configs" / "experiment.yaml")
    profiles = deepcopy(base_config.profiles)
    profiles["smoke"]["fedcpdp"]["outer_seed_batch_size"] = 2
    seeds = base_config.seeds[:2]
    config = replace(
        base_config,
        output_dir=tmp_path,
        repetitions=2,
        seeds=seeds,
        profiles=profiles,
    )

    report = ExperimentRunner(config).run(
        ["fedcpdp"],
        profile="smoke",
        targets=["ReLink/Apache"],
        seeds=seeds,
        workers=1,
        verbose=False,
    )

    assert report.succeeded == 2
    assert report.failed == 0
    for seed in seeds:
        result_path = (
            tmp_path
            / "runs"
            / "smoke"
            / "fedcpdp"
            / "ReLink"
            / "Apache"
            / f"seed_{seed}.json"
        )
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        assert payload["status"] == "success"
        assert payload["execution"]["seed_batch_size"] == 2
        assert payload["method_metadata"]["outer_seed_batch_size"] == 2
