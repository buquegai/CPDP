from __future__ import annotations

import hashlib
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple, Union

import numpy as np

from cpdp.methods.base import BaseMethod, MethodProvenance
from cpdp.methods.fedcpdp_torch import TorchFedCPDPCore
from cpdp.types import (
    LabeledProject,
    PredictionResult,
    ProjectFeatures,
    UnlabeledFeatures,
    UnlabeledProject,
)
from cpdp.vendor import load_module
from threadpoolctl import threadpool_limits


def _stable_hash_index(name: str, dimension: int) -> int:
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % dimension


def _hash_features(
    features: Union[ProjectFeatures, UnlabeledFeatures], dimension: int
) -> np.ndarray:
    result = np.zeros((features.n_samples, dimension), dtype=np.float32)
    for feature_index, feature_name in enumerate(features.feature_names):
        if feature_name == features.effort_name:
            continue
        column = features.values[:, feature_index].astype(np.float64, copy=True)
        mean = float(np.nanmean(column))
        standard_deviation = float(np.nanstd(column))
        if not np.isfinite(standard_deviation) or standard_deviation == 0:
            standard_deviation = 1.0
        column = (column - mean) / standard_deviation
        column = np.nan_to_num(column, nan=0.0, posinf=0.0, neginf=0.0)
        result[:, _stable_hash_index(feature_name, dimension)] += column.astype(np.float32)
    return result


def _train_internal_seed(
    official,
    clients: Sequence[Dict[str, Any]],
    internal_seed: int,
) -> Tuple[Any, Any, Sequence[np.ndarray]]:
    parameters, server_state = official.train_one_seed(clients, internal_seed)
    source_scores = [
        official.model_predict_scores(parameters, client["X"]) for client in clients
    ]
    return parameters, server_state, source_scores


def _ensemble_seeds(seed: int, size: int, strategy: str) -> Sequence[int]:
    normalized = str(strategy).strip().lower()
    if normalized == "legacy_overlap":
        return [int(seed + offset) for offset in range(size)]
    if normalized != "independent_blake2":
        raise ValueError(
            "FedCPDP seed_strategy must be 'legacy_overlap' or 'independent_blake2'"
        )
    output = []
    used = set()
    for offset in range(size):
        attempt = 0
        while True:
            payload = f"fedcpdp-v1:{int(seed)}:{offset}:{attempt}".encode("ascii")
            candidate = int.from_bytes(
                hashlib.blake2b(payload, digest_size=4).digest(), "little"
            )
            if candidate not in used:
                used.add(candidate)
                output.append(candidate)
                break
            attempt += 1
    return output


class FedCPDPMethod(BaseMethod):
    name = "fedcpdp"
    provenance = MethodProvenance(
        display_name="FedCPDP",
        implementation="Official NumPy core with a manual-equation Torch/CUDA backend",
        fidelity="official-core-adapter-and-equivalent-gpu-port",
        source_url="https://github.com/MeeteshNevendra/FedCPDP",
        source_revision="ab5af63 (GitHub archive snapshot)",
        target_label_policy="Held-out target labels are used only by the common evaluator",
        notes=(
            "Feature hashing, MLP, FedProx, focal loss, representation alignment, adaptive "
            "aggregation, and source-only threshold selection preserve the official equations. "
            "Python's process-randomized hash() is replaced by deterministic BLAKE2 hashing to "
            "make the intended fixed feature map reproducible."
        ),
    )

    def _official_module(self):
        roots = tuple(
            (self.project_root / "third_party" / "original" / "FedCPDP").glob(
                "MeeteshNevendra-FedCPDP-*"
            )
        )
        if len(roots) != 1:
            raise RuntimeError(f"Expected one FedCPDP source snapshot, found {len(roots)}")
        original_directory = Path.cwd()
        with tempfile.TemporaryDirectory(prefix="cpdp_fedcpdp_import_") as temporary:
            try:
                os.chdir(temporary)
                return load_module(
                    "cpdp_vendor_fedcpdp", roots[0] / "src" / "fed_cpdp.py"
                )
            finally:
                os.chdir(original_directory)

    def fit_predict(
        self,
        sources: Sequence[LabeledProject],
        target: UnlabeledProject,
        seed: int,
    ) -> PredictionResult:
        if not sources:
            raise ValueError("FedCPDP requires at least one source client")
        official = self._official_module()
        hash_dimension = int(self.params.get("hash_dim", 1024))
        official.HASH_DIM = hash_dimension
        official.ROUNDS = int(self.params.get("rounds", 80))
        official.LOCAL_EPOCHS = int(self.params.get("local_epochs", 12))
        official.GLOBAL_BASE_SEED = int(seed)

        ensemble_size = int(self.params.get("ensemble_size", 3))
        if ensemble_size < 1:
            raise ValueError("FedCPDP ensemble_size must be at least 1")
        seed_strategy = str(self.params.get("seed_strategy", "legacy_overlap"))
        ensemble_seeds = list(_ensemble_seeds(seed, ensemble_size, seed_strategy))
        official.ENSEMBLE_SEEDS = ensemble_seeds
        backend = str(self.params.get("backend", "numpy")).strip().lower()
        if backend not in {"numpy", "torch"}:
            raise ValueError("FedCPDP backend must be 'numpy' or 'torch'")
        requested_workers = int(self.params.get("parallel_workers", ensemble_size))
        if requested_workers < 1:
            raise ValueError("FedCPDP parallel_workers must be at least 1")
        parallel_workers = (
            min(requested_workers, ensemble_size) if backend == "numpy" else 1
        )

        clients = [
            {
                "name": source.key,
                "X": _hash_features(source.features, hash_dimension),
                "y": source.labels.copy(),
                "effort": source.features.effort.copy(),
            }
            for source in sources
        ]
        target_x = _hash_features(target.features, hash_dimension)

        source_scores_by_seed = []
        runtime_metadata: Dict[str, Any]
        if backend == "numpy":
            results = [None] * ensemble_size
            configured_blas_threads = int(self.params.get("blas_threads", 0))
            blas_threads = configured_blas_threads or max(
                1, int(os.cpu_count() or 1) // parallel_workers
            )
            with threadpool_limits(limits=blas_threads):
                if parallel_workers == 1:
                    for index, internal_seed in enumerate(ensemble_seeds):
                        results[index] = _train_internal_seed(
                            official, clients, internal_seed
                        )
                        print(
                            f"PROGRESS fedcpdp internal_model={index + 1}/{ensemble_size} complete",
                            flush=True,
                        )
                else:
                    with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
                        future_indices = {
                            executor.submit(
                                _train_internal_seed, official, clients, internal_seed
                            ): index
                            for index, internal_seed in enumerate(ensemble_seeds)
                        }
                        for future in as_completed(future_indices):
                            index = future_indices[future]
                            results[index] = future.result()
                            print(
                                f"PROGRESS fedcpdp internal_model={index + 1}/{ensemble_size} complete",
                                flush=True,
                            )
            completed_results = [result for result in results if result is not None]
            if len(completed_results) != ensemble_size:
                raise RuntimeError("FedCPDP did not complete every internal ensemble model")
            models = [
                (parameters, server_state)
                for parameters, server_state, _ in completed_results
            ]
            source_scores_by_seed = [
                source_scores for _, _, source_scores in completed_results
            ]
            target_scores = np.mean(
                np.stack(
                    [
                        official.model_predict_scores(parameters, target_x)
                        for parameters, _ in models
                    ],
                    axis=0,
                ),
                axis=0,
            )
            runtime_metadata = {
                "backend": "numpy",
                "device": "cpu",
                "dtype": "float64",
                "blas_threads_per_worker": blas_threads,
            }
        else:
            core = TorchFedCPDPCore(
                official,
                device=str(self.params.get("device", "auto")),
                dtype=str(self.params.get("dtype", "float32")),
            )
            torch_clients = core.prepare_clients(clients)
            target_tensor = core.prepare_values(target_x)
            try:
                batch_ensemble = bool(
                    self.params.get("batch_ensemble_on_device", True)
                )
                if batch_ensemble:
                    parameters, _ = core.train_ensemble(
                        torch_clients, ensemble_seeds
                    )
                    source_score_matrices = [
                        core.predict_ensemble_scores(parameters, client["X"])
                        for client in torch_clients
                    ]
                    source_scores_by_seed = [
                        [
                            source_score_matrices[client_index][model_index]
                            for client_index in range(len(torch_clients))
                        ]
                        for model_index in range(ensemble_size)
                    ]
                    target_scores = np.mean(
                        core.predict_ensemble_scores(parameters, target_tensor),
                        axis=0,
                    )
                    print(
                        "PROGRESS fedcpdp "
                        f"internal_models={ensemble_size}/{ensemble_size} "
                        "complete mode=batched",
                        flush=True,
                    )
                else:
                    torch_results = []
                    for index, internal_seed in enumerate(ensemble_seeds):
                        parameters, server_state = core.train_one_seed(
                            torch_clients, internal_seed
                        )
                        source_scores = [
                            core.predict_scores(parameters, client["X"])
                            for client in torch_clients
                        ]
                        torch_results.append(
                            (parameters, server_state, source_scores)
                        )
                        print(
                            f"PROGRESS fedcpdp internal_model={index + 1}/{ensemble_size} complete",
                            flush=True,
                        )
                    models = [
                        (model_parameters, server_state)
                        for model_parameters, server_state, _ in torch_results
                    ]
                    source_scores_by_seed = [
                        source_scores for _, _, source_scores in torch_results
                    ]
                    target_scores = np.mean(
                        np.stack(
                            [
                                core.predict_scores(
                                    model_parameters, target_tensor
                                )
                                for model_parameters, _ in models
                            ],
                            axis=0,
                        ),
                        axis=0,
                    )
                import torch

                runtime_metadata = {
                    "backend": "torch-manual",
                    "device": str(core.device),
                    "dtype": str(core.dtype).replace("torch.", ""),
                    "batch_ensemble_on_device": batch_ensemble,
                    "cuda_device_name": (
                        torch.cuda.get_device_name(core.device)
                        if core.device.type == "cuda"
                        else None
                    ),
                    "blas_threads_per_worker": None,
                }
            finally:
                core.release()

        source_ensemble_scores = []
        source_labels = []
        source_efforts = []
        for client_index, client in enumerate(clients):
            source_ensemble_scores.append(
                np.mean(
                    np.stack(
                        [scores[client_index] for scores in source_scores_by_seed], axis=0
                    ),
                    axis=0,
                )
            )
            source_labels.append(client["y"])
            source_efforts.append(client["effort"])
        threshold = official.find_best_threshold_from_scores(
            source_ensemble_scores,
            source_labels,
            source_efforts,
            mode=official.THR_MODE,
            pmin=official.P_MIN,
        )

        return PredictionResult(
            scores=target_scores,
            threshold=float(threshold),
            metadata={
                "source_clients": [source.key for source in sources],
                "rounds": official.ROUNDS,
                "local_epochs": official.LOCAL_EPOCHS,
                "ensemble_seeds": ensemble_seeds,
                "seed_strategy": seed_strategy,
                "parallel_workers": parallel_workers,
                "hash_dim": hash_dimension,
                "hash_function": "BLAKE2b-64",
                "imbalance_handling": "official focal loss / weighted BCE path",
                "aggregator": official.AGGREGATOR,
                "fedprox": bool(official.USE_FEDPROX),
                "differential_privacy": bool(official.USE_DP),
                "result_signature": self.params.get("result_signature"),
                **runtime_metadata,
            },
        )

    def fit_predict_many(
        self,
        sources: Sequence[LabeledProject],
        target: UnlabeledProject,
        seeds: Sequence[int],
    ) -> Dict[int, PredictionResult]:
        outer_seeds = tuple(int(seed) for seed in seeds)
        if not outer_seeds:
            raise ValueError("FedCPDP requires at least one outer seed")
        if len(set(outer_seeds)) != len(outer_seeds):
            raise ValueError("FedCPDP outer seeds must be unique")
        if len(outer_seeds) == 1:
            seed = outer_seeds[0]
            return {seed: self.fit_predict(sources, target, seed)}

        backend = str(self.params.get("backend", "numpy")).strip().lower()
        batch_ensemble = bool(self.params.get("batch_ensemble_on_device", True))
        if backend != "torch" or not batch_ensemble:
            return {
                seed: self.fit_predict(sources, target, seed) for seed in outer_seeds
            }
        if not sources:
            raise ValueError("FedCPDP requires at least one source client")

        official = self._official_module()
        hash_dimension = int(self.params.get("hash_dim", 1024))
        official.HASH_DIM = hash_dimension
        official.ROUNDS = int(self.params.get("rounds", 80))
        official.LOCAL_EPOCHS = int(self.params.get("local_epochs", 12))
        official.GLOBAL_BASE_SEED = outer_seeds[0]

        ensemble_size = int(self.params.get("ensemble_size", 3))
        if ensemble_size < 1:
            raise ValueError("FedCPDP ensemble_size must be at least 1")
        seed_strategy = str(self.params.get("seed_strategy", "legacy_overlap"))
        grouped_ensemble_seeds = {
            seed: tuple(_ensemble_seeds(seed, ensemble_size, seed_strategy))
            for seed in outer_seeds
        }
        flattened_ensemble_seeds = tuple(
            internal_seed
            for seed in outer_seeds
            for internal_seed in grouped_ensemble_seeds[seed]
        )
        if len(set(flattened_ensemble_seeds)) != len(flattened_ensemble_seeds):
            raise ValueError(
                "FedCPDP batched outer seeds produced overlapping internal seeds"
            )
        official.ENSEMBLE_SEEDS = list(flattened_ensemble_seeds)

        clients = [
            {
                "name": source.key,
                "X": _hash_features(source.features, hash_dimension),
                "y": source.labels.copy(),
                "effort": source.features.effort.copy(),
            }
            for source in sources
        ]
        target_x = _hash_features(target.features, hash_dimension)
        core = TorchFedCPDPCore(
            official,
            device=str(self.params.get("device", "auto")),
            dtype=str(self.params.get("dtype", "float32")),
        )
        torch_clients = core.prepare_clients(clients)
        target_tensor = core.prepare_values(target_x)
        try:
            parameters, _ = core.train_ensemble(
                torch_clients, flattened_ensemble_seeds
            )
            source_score_matrices = [
                core.predict_ensemble_scores(parameters, client["X"])
                for client in torch_clients
            ]
            target_score_matrix = core.predict_ensemble_scores(
                parameters, target_tensor
            )
            import torch

            runtime_metadata = {
                "backend": "torch-manual",
                "device": str(core.device),
                "dtype": str(core.dtype).replace("torch.", ""),
                "batch_ensemble_on_device": True,
                "outer_seed_batch_size": len(outer_seeds),
                "models_in_device_batch": len(flattened_ensemble_seeds),
                "cuda_device_name": (
                    torch.cuda.get_device_name(core.device)
                    if core.device.type == "cuda"
                    else None
                ),
                "blas_threads_per_worker": None,
            }
        finally:
            core.release()

        outputs = {}
        source_labels = [client["y"] for client in clients]
        source_efforts = [client["effort"] for client in clients]
        for outer_index, seed in enumerate(outer_seeds):
            start = outer_index * ensemble_size
            stop = start + ensemble_size
            source_ensemble_scores = [
                np.mean(scores[start:stop], axis=0)
                for scores in source_score_matrices
            ]
            target_scores = np.mean(target_score_matrix[start:stop], axis=0)
            threshold = official.find_best_threshold_from_scores(
                source_ensemble_scores,
                source_labels,
                source_efforts,
                mode=official.THR_MODE,
                pmin=official.P_MIN,
            )
            outputs[seed] = PredictionResult(
                scores=target_scores,
                threshold=float(threshold),
                metadata={
                    "source_clients": [source.key for source in sources],
                    "rounds": official.ROUNDS,
                    "local_epochs": official.LOCAL_EPOCHS,
                    "ensemble_seeds": list(grouped_ensemble_seeds[seed]),
                    "seed_strategy": seed_strategy,
                    "parallel_workers": 1,
                    "hash_dim": hash_dimension,
                    "hash_function": "BLAKE2b-64",
                    "imbalance_handling": (
                        "official focal loss / weighted BCE path"
                    ),
                    "aggregator": official.AGGREGATOR,
                    "fedprox": bool(official.USE_FEDPROX),
                    "differential_privacy": bool(official.USE_DP),
                    "result_signature": self.params.get("result_signature"),
                    **runtime_metadata,
                },
            )

        print(
            "PROGRESS fedcpdp "
            f"outer_seeds={len(outer_seeds)}/{len(outer_seeds)} "
            f"internal_models={len(flattened_ensemble_seeds)} complete "
            "mode=outer-seed-batched",
            flush=True,
        )
        return outputs
