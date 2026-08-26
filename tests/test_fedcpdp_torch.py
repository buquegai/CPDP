from pathlib import Path

import numpy as np
import pytest

from cpdp.config import load_config
from cpdp.methods.fedcpdp import FedCPDPMethod, _ensemble_seeds
from cpdp.methods.fedcpdp_torch import TorchFedCPDPCore
from cpdp.runner import (
    RunTask,
    _method_uses_exclusive_cuda,
    _sequential_task_groups,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_independent_internal_seeds_do_not_overlap_between_outer_runs() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "experiment.yaml")
    groups = [
        set(_ensemble_seeds(seed, 3, "independent_blake2"))
        for seed in config.seeds
    ]

    assert all(len(group) == 3 for group in groups)
    assert len(set().union(*groups)) == 3 * len(groups)
    assert max(set().union(*groups)) <= np.iinfo(np.uint32).max


def test_torch_manual_loss_and_grads_match_official_numpy_without_dropout() -> None:
    torch = pytest.importorskip("torch")
    method = FedCPDPMethod(PROJECT_ROOT, {})
    official = method._official_module()
    official.HASH_DIM = 4
    official.H1_DIM = 3
    official.H2_DIM = 2
    official.DROPOUT_P = 0.0

    seed = 37
    model = official.MLP2(
        official.HASH_DIM,
        official.H1_DIM,
        official.H2_DIM,
        1,
        l2=official.L2_WEIGHT_DECAY,
        seed=seed,
    )
    parameters = model.get_params()
    global_parameters = tuple(value + 0.01 for value in parameters)
    values = np.asarray(
        [
            [0.2, -0.3, 0.7, 1.1],
            [-0.4, 0.8, 0.1, -0.2],
            [1.0, 0.3, -0.6, 0.4],
            [-0.7, -0.1, 0.2, 0.9],
        ],
        dtype=np.float64,
    )
    labels = np.asarray([0, 1, 0, 1], dtype=np.float64)
    global_mean = np.asarray([0.15, -0.05], dtype=np.float64)
    global_variance = np.asarray([0.8, 1.2], dtype=np.float64)

    numpy_loss, numpy_grads, numpy_mean, numpy_variance = model.loss_and_grads(
        values,
        labels,
        global_params=global_parameters,
        fedprox_mu=official.FEDPROX_MU,
        align_mean_w=official.ALIGN_MEAN_WEIGHT,
        align_var_w=official.ALIGN_VAR_WEIGHT,
        global_mean=global_mean,
        global_var=global_variance,
        use_focal=True,
        focal_gamma=official.FOCAL_GAMMA,
        dropout_p=0.0,
        rs=np.random.RandomState(seed),
    )

    core = TorchFedCPDPCore(official, device="cpu", dtype="float64")
    torch_parameters = tuple(
        torch.as_tensor(value, dtype=torch.float64) for value in parameters
    )
    torch_global_parameters = tuple(
        torch.as_tensor(value, dtype=torch.float64) for value in global_parameters
    )
    torch_loss, torch_grads, torch_mean, torch_variance = core.loss_and_grads(
        torch_parameters,
        torch.as_tensor(values, dtype=torch.float64),
        torch.as_tensor(labels, dtype=torch.float64),
        global_parameters=torch_global_parameters,
        fedprox_mu=official.FEDPROX_MU,
        align_mean_weight=official.ALIGN_MEAN_WEIGHT,
        align_variance_weight=official.ALIGN_VAR_WEIGHT,
        global_mean=torch.as_tensor(global_mean, dtype=torch.float64),
        global_variance=torch.as_tensor(global_variance, dtype=torch.float64),
        use_focal=True,
        focal_gamma=official.FOCAL_GAMMA,
        dropout_probability=0.0,
        generator=None,
    )

    assert torch_loss.item() == pytest.approx(numpy_loss, rel=1e-12, abs=1e-12)
    for numpy_grad, torch_grad in zip(numpy_grads, torch_grads):
        np.testing.assert_allclose(
            torch_grad.numpy(), numpy_grad, rtol=1e-12, atol=1e-12
        )
    np.testing.assert_allclose(torch_mean.numpy(), numpy_mean, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        torch_variance.numpy(), numpy_variance, rtol=1e-12, atol=1e-12
    )


def test_batched_torch_ensemble_matches_sequential_training() -> None:
    torch = pytest.importorskip("torch")
    method = FedCPDPMethod(PROJECT_ROOT, {})
    official = method._official_module()
    official.HASH_DIM = 4
    official.H1_DIM = 3
    official.H2_DIM = 2
    official.DROPOUT_P = 0.2
    official.ROUNDS = 2
    official.LOCAL_EPOCHS = 2
    official.BATCH_SIZE = 3
    official.USE_DP = False

    core = TorchFedCPDPCore(official, device="cpu", dtype="float64")
    clients = core.prepare_clients(
        [
            {
                "name": "source-a",
                "X": np.asarray(
                    [
                        [0.2, -0.3, 0.7, 1.1],
                        [-0.4, 0.8, 0.1, -0.2],
                        [1.0, 0.3, -0.6, 0.4],
                        [-0.7, -0.1, 0.2, 0.9],
                        [0.6, 0.5, -0.4, -0.8],
                    ],
                    dtype=np.float64,
                ),
                "y": np.asarray([0, 1, 0, 1, 1], dtype=np.float64),
                "effort": np.ones(5, dtype=np.float64),
            },
            {
                "name": "source-b",
                "X": np.asarray(
                    [
                        [-0.1, 0.4, 0.3, -0.5],
                        [0.9, -0.2, 0.6, 0.7],
                        [-0.8, 0.1, -0.3, 0.2],
                        [0.5, 0.6, 0.2, -0.4],
                    ],
                    dtype=np.float64,
                ),
                "y": np.asarray([1, 0, 0, 1], dtype=np.float64),
                "effort": np.ones(4, dtype=np.float64),
            },
        ]
    )
    seeds = [11, 29, 47]

    sequential = [core.train_one_seed(clients, seed) for seed in seeds]
    batched_parameters, batched_state = core.train_ensemble(clients, seeds)

    for model_index, (parameters, server_state) in enumerate(sequential):
        for sequential_value, batched_value in zip(parameters, batched_parameters):
            torch.testing.assert_close(
                batched_value[model_index], sequential_value, rtol=1e-11, atol=1e-11
            )
        for state_name in ("global_mean", "global_var", "momentum"):
            torch.testing.assert_close(
                batched_state[state_name][model_index],
                server_state[state_name],
                rtol=1e-11,
                atol=1e-11,
            )

    probe = clients[0]["X"]
    sequential_scores = np.stack(
        [core.predict_scores(parameters, probe) for parameters, _ in sequential],
        axis=0,
    )
    batched_scores = core.predict_ensemble_scores(batched_parameters, probe)
    np.testing.assert_allclose(
        batched_scores, sequential_scores, rtol=1e-11, atol=1e-11
    )


def test_full_fedcpdp_uses_exclusive_cuda_when_available() -> None:
    torch = pytest.importorskip("torch")
    config = load_config(PROJECT_ROOT / "configs" / "experiment.yaml")

    assert _method_uses_exclusive_cuda(config, "full", "fedcpdp") is bool(
        torch.cuda.is_available()
    )


def test_fedcpdp_outer_seeds_are_grouped_without_crossing_targets(
    monkeypatch,
) -> None:
    import cpdp.runner as runner_module

    config = load_config(PROJECT_ROOT / "configs" / "experiment.yaml")
    monkeypatch.setattr(
        runner_module,
        "_method_uses_exclusive_cuda",
        lambda *_args, **_kwargs: True,
    )
    tasks = tuple(
        RunTask(
            method="fedcpdp",
            family="NASA",
            target="CM1",
            seed=seed,
            profile="full",
            force=False,
        )
        for seed in range(12)
    ) + (
        RunTask(
            method="fedcpdp",
            family="NASA",
            target="MW1",
            seed=0,
            profile="full",
            force=False,
        ),
    )

    groups = _sequential_task_groups(config, tasks)

    assert [len(group) for group in groups] == [10, 2, 1]
    assert all(task.target == "CM1" for task in groups[0] + groups[1])
    assert groups[2][0].target == "MW1"
