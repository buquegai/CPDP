from __future__ import annotations

import copy
import multiprocessing as mp
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from imblearn.over_sampling import SMOTE
from scipy.special import expit
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.linear_model import (
    LogisticRegression,
    PassiveAggressiveClassifier,
    Perceptron,
    RidgeClassifier,
)
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from threadpoolctl import threadpool_limits

from cpdp.methods.base import BaseMethod, MethodProvenance
from cpdp.types import LabeledProject, PredictionResult, UnlabeledProject


ADAPTATIONS = ("none", "nnfilter", "cde_smote")
FEATURE_SELECTORS = ("none", "lasso", "pca", "rf_importance", "variance")
CLASSIFIERS = (
    "rf",
    "knn",
    "svm",
    "lr",
    "dt",
    "nb",
    "ridge",
    "pac",
    "perceptron",
    "mlp",
    "extra_tree",
    "adaboost",
    "bagging",
    "extra_trees",
)


class HardTimeoutError(TimeoutError):
    pass


class SubprocessEvaluationError(RuntimeError):
    pass


def _subprocess_entry(
    send_connection: Any,
    function: Callable[..., Any],
    arguments: Tuple[Any, ...],
) -> None:
    try:
        send_connection.send(("success", function(*arguments)))
    except BaseException as exc:
        send_connection.send(("failure", type(exc).__name__, str(exc)))
    finally:
        send_connection.close()


def _stop_process(process: mp.Process) -> None:
    if process.is_alive():
        process.terminate()
        process.join(timeout=1.0)
    if process.is_alive() and hasattr(process, "kill"):
        process.kill()
        process.join(timeout=1.0)


def _call_with_hard_timeout(
    function: Callable[..., Any],
    arguments: Tuple[Any, ...],
    timeout_seconds: float,
) -> Any:
    if timeout_seconds <= 0:
        raise ValueError("Hard timeout must be positive")

    context = mp.get_context("spawn")
    receive_connection, send_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_subprocess_entry,
        args=(send_connection, function, arguments),
    )
    process.start()
    send_connection.close()

    try:
        if not receive_connection.poll(timeout_seconds):
            raise HardTimeoutError(
                f"Subprocess exceeded the {timeout_seconds:.1f}s hard timeout"
            )
        try:
            payload = receive_connection.recv()
        except EOFError as exc:
            raise SubprocessEvaluationError(
                f"Subprocess exited without returning a result (exitcode={process.exitcode})"
            ) from exc
        process.join(timeout=1.0)
        if payload[0] == "failure":
            raise SubprocessEvaluationError(f"{payload[1]}: {payload[2]}")
        return payload[1]
    finally:
        _stop_process(process)
        receive_connection.close()
        process.close()


@dataclass
class Candidate:
    adaptation: str
    feature_selector: str
    classifier: str
    parameters: Dict[str, Any]
    auc: float
    f1: float


class FeatureTransform:
    def __init__(self, name: str, parameters: Dict[str, Any], seed: int) -> None:
        self.name = name
        self.parameters = parameters
        self.seed = seed
        self.scaler = None
        self.transformer = None

    def fit_transform(self, source_x: np.ndarray, source_y: np.ndarray) -> np.ndarray:
        if self.name == "none":
            return source_x
        if self.name == "pca":
            components = min(
                int(self.parameters["pca_components"]), source_x.shape[1], source_x.shape[0]
            )
            self.transformer = PCA(n_components=max(1, components), random_state=self.seed)
            return self.transformer.fit_transform(source_x)
        if self.name == "lasso":
            self.scaler = RobustScaler().fit(source_x)
            scaled = self.scaler.transform(source_x)
            selector_model = LogisticRegression(
                C=float(self.parameters["lasso_c"]),
                penalty="l1",
                solver="liblinear",
                random_state=self.seed,
                max_iter=500,
            )
            selector_model.fit(scaled, source_y)
            self.transformer = SelectFromModel(selector_model, prefit=True)
            transformed = self.transformer.transform(scaled)
            return transformed if transformed.shape[1] else scaled
        if self.name == "rf_importance":
            selector_model = RandomForestClassifier(
                n_estimators=int(self.parameters["fs_trees"]),
                max_depth=int(self.parameters["fs_depth"]),
                n_jobs=1,
                random_state=self.seed,
            ).fit(source_x, source_y)
            self.transformer = SelectFromModel(
                selector_model, threshold=self.parameters["fs_threshold"], prefit=True
            )
            transformed = self.transformer.transform(source_x)
            return transformed if transformed.shape[1] else source_x
        if self.name == "variance":
            threshold = float(self.parameters["variance_threshold"])
            frequencies = []
            for column in source_x.T:
                _, counts = np.unique(column, return_counts=True)
                frequencies.append(float(np.max(counts) / len(column)))
            keep = np.asarray(frequencies) < threshold
            if not np.any(keep):
                keep[:] = True
            self.transformer = keep
            return source_x[:, keep]
        raise ValueError(f"Unsupported feature selector: {self.name}")

    def transform(self, target_x: np.ndarray) -> np.ndarray:
        if self.name == "none":
            return target_x
        if self.name == "pca":
            return self.transformer.transform(target_x)
        if self.name == "lasso":
            scaled = self.scaler.transform(target_x)
            transformed = self.transformer.transform(scaled)
            return transformed if transformed.shape[1] else scaled
        if self.name == "rf_importance":
            transformed = self.transformer.transform(target_x)
            return transformed if transformed.shape[1] else target_x
        if self.name == "variance":
            return target_x[:, self.transformer]
        raise ValueError(f"Unsupported feature selector: {self.name}")


def _nnfilter(
    source_x: np.ndarray,
    source_y: np.ndarray,
    target_x: np.ndarray,
    neighbors: int,
    metric: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_log = np.log1p(np.maximum(source_x, 0))
    target_log = np.log1p(np.maximum(target_x, 0))
    count = min(max(1, neighbors), len(source_log))
    model = NearestNeighbors(metric=metric).fit(source_log)
    selected = np.unique(model.kneighbors(target_log, count, return_distance=False).reshape(-1))
    return source_log[selected], source_y[selected], target_log


def _scores(model: BaseEstimator, values: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(values)
        if probabilities.ndim == 2 and probabilities.shape[1] > 1:
            return probabilities[:, 1].astype(np.float64)
    if hasattr(model, "decision_function"):
        return expit(np.asarray(model.decision_function(values), dtype=np.float64).reshape(-1))
    return np.asarray(model.predict(values), dtype=np.float64).reshape(-1)


def _make_classifier(name: str, parameters: Dict[str, Any], seed: int) -> BaseEstimator:
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=int(parameters["trees"]),
            criterion=parameters["criterion"],
            max_features=float(parameters["max_features"]),
            min_samples_split=int(parameters["min_split"]),
            min_samples_leaf=int(parameters["min_leaf"]),
            random_state=seed,
            n_jobs=1,
        )
    if name == "knn":
        return KNeighborsClassifier(
            n_neighbors=int(parameters["neighbors"]), p=int(parameters["p"])
        )
    if name == "svm":
        return SVC(
            kernel=parameters["kernel"],
            C=float(parameters["c"]),
            gamma=parameters["gamma"],
            probability=True,
            random_state=seed,
        )
    if name == "lr":
        return LogisticRegression(
            C=float(parameters["c"]),
            penalty=parameters["penalty"],
            solver="liblinear",
            fit_intercept=bool(parameters["fit_intercept"]),
            max_iter=int(parameters["max_iter"]),
            random_state=seed,
        )
    if name == "dt":
        return DecisionTreeClassifier(
            criterion=parameters["criterion"],
            splitter=parameters["splitter"],
            max_features=float(parameters["max_features"]),
            min_samples_split=int(parameters["min_split"]),
            min_samples_leaf=int(parameters["min_leaf"]),
            random_state=seed,
        )
    if name == "nb":
        return GaussianNB()
    if name == "ridge":
        return RidgeClassifier(
            alpha=float(parameters["alpha"]),
            fit_intercept=bool(parameters["fit_intercept"]),
            tol=float(parameters["tol"]),
        )
    if name == "pac":
        return PassiveAggressiveClassifier(
            C=float(parameters["c"]),
            fit_intercept=bool(parameters["fit_intercept"]),
            tol=float(parameters["tol"]),
            loss=parameters["loss"],
            random_state=seed,
        )
    if name == "perceptron":
        return Perceptron(
            penalty=parameters["penalty"],
            alpha=float(parameters["alpha"]),
            fit_intercept=bool(parameters["fit_intercept"]),
            tol=float(parameters["tol"]),
            random_state=seed,
        )
    if name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(int(parameters["hidden"]),),
            activation=parameters["activation"],
            max_iter=int(parameters["max_iter"]),
            solver=parameters["solver"],
            random_state=seed,
        )
    if name == "extra_tree":
        return ExtraTreeClassifier(
            criterion=parameters["criterion"],
            splitter=parameters["splitter"],
            max_features=float(parameters["max_features"]),
            min_samples_split=int(parameters["min_split"]),
            min_samples_leaf=int(parameters["min_leaf"]),
            random_state=seed,
        )
    if name == "adaboost":
        return AdaBoostClassifier(
            n_estimators=int(parameters["trees"]),
            learning_rate=float(parameters["learning_rate"]),
            random_state=seed,
        )
    if name == "bagging":
        return BaggingClassifier(
            n_estimators=int(parameters["trees"]),
            max_samples=float(parameters["max_samples"]),
            max_features=float(parameters["max_features"]),
            random_state=seed,
            n_jobs=1,
        )
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=int(parameters["trees"]),
            criterion=parameters["criterion"],
            max_features=float(parameters["max_features"]),
            min_samples_split=int(parameters["min_split"]),
            min_samples_leaf=int(parameters["min_leaf"]),
            random_state=seed,
            n_jobs=1,
        )
    raise ValueError(f"Unsupported classifier: {name}")


def _sample_parameters(
    rng: np.random.Generator, feature_selector: str, classifier: str
) -> Dict[str, Any]:
    parameters: Dict[str, Any] = {
        "nn_neighbors": int(rng.integers(1, 101)),
        "nn_metric": str(rng.choice(["cityblock", "cosine", "euclidean", "manhattan"])),
        "cde_neighbors": int(rng.integers(1, 11)),
    }
    if feature_selector == "pca":
        parameters["pca_components"] = int(rng.integers(1, 21))
    elif feature_selector == "lasso":
        parameters["lasso_c"] = float(rng.uniform(0.1, 1.0))
    elif feature_selector == "rf_importance":
        parameters.update(
            fs_trees=int(rng.integers(10, 201)),
            fs_depth=int(rng.integers(10, 201)),
            fs_threshold=str(rng.choice(["mean", "median"])),
        )
    elif feature_selector == "variance":
        parameters["variance_threshold"] = float(rng.uniform(0.6, 0.9))

    if classifier in {"rf", "extra_trees"}:
        parameters.update(
            trees=int(rng.integers(10, 201)),
            criterion=str(rng.choice(["gini", "entropy"])),
            max_features=float(rng.uniform(0.2, 1.0)),
            min_split=int(rng.integers(2, 41)),
            min_leaf=int(rng.integers(1, 21)),
        )
    elif classifier == "knn":
        parameters.update(neighbors=int(rng.integers(1, 11)), p=int(rng.integers(1, 5)))
    elif classifier == "svm":
        parameters.update(
            kernel=str(rng.choice(["linear", "rbf", "sigmoid"])),
            c=float(rng.uniform(0.001, 10.0)),
            gamma=float(10 ** rng.uniform(-2, 2)),
        )
    elif classifier == "lr":
        parameters.update(
            c=float(rng.uniform(0.001, 10.0)),
            penalty=str(rng.choice(["l1", "l2"])),
            fit_intercept=bool(rng.integers(0, 2)),
            max_iter=int(rng.integers(50, 201)),
        )
    elif classifier in {"dt", "extra_tree"}:
        parameters.update(
            criterion=str(rng.choice(["gini", "entropy"])),
            splitter=str(rng.choice(["best", "random"])),
            max_features=float(rng.uniform(0.2, 1.0)),
            min_split=int(rng.integers(2, 41)),
            min_leaf=int(rng.integers(1, 21)),
        )
    elif classifier == "ridge":
        parameters.update(
            alpha=float(10 ** rng.uniform(-3, 2)),
            fit_intercept=bool(rng.integers(0, 2)),
            tol=float(10 ** rng.uniform(-5, -1)),
        )
    elif classifier == "pac":
        parameters.update(
            c=float(10 ** rng.uniform(-3, 2)),
            fit_intercept=bool(rng.integers(0, 2)),
            tol=float(10 ** rng.uniform(-5, -1)),
            loss=str(rng.choice(["hinge", "squared_hinge"])),
        )
    elif classifier == "perceptron":
        parameters.update(
            penalty=str(rng.choice(["l1", "l2"])),
            alpha=float(10 ** rng.uniform(-5, -1)),
            fit_intercept=bool(rng.integers(0, 2)),
            tol=float(10 ** rng.uniform(-5, -1)),
        )
    elif classifier == "mlp":
        parameters.update(
            hidden=int(rng.integers(50, 201)),
            activation=str(rng.choice(["identity", "logistic", "tanh", "relu"])),
            max_iter=int(rng.integers(100, 251)),
            solver=str(rng.choice(["lbfgs", "sgd", "adam"])),
        )
    elif classifier == "adaboost":
        parameters.update(
            trees=int(rng.integers(10, 201)),
            learning_rate=float(10 ** rng.uniform(-2, 1)),
        )
    elif classifier == "bagging":
        parameters.update(
            trees=int(rng.integers(10, 201)),
            max_samples=float(rng.uniform(0.7, 1.0)),
            max_features=float(rng.uniform(0.7, 1.0)),
        )
    return parameters


def _fit_pipeline(
    adaptation: str,
    feature_selector: str,
    classifier: str,
    parameters: Dict[str, Any],
    source_x: np.ndarray,
    source_y: np.ndarray,
    target_x: np.ndarray,
    seed: int,
) -> Tuple[BaseEstimator, FeatureTransform, np.ndarray]:
    selected_source_x = source_x.copy()
    selected_source_y = source_y.copy()
    selected_target_x = target_x.copy()
    if adaptation == "nnfilter":
        selected_source_x, selected_source_y, selected_target_x = _nnfilter(
            selected_source_x,
            selected_source_y,
            selected_target_x,
            parameters["nn_neighbors"],
            parameters["nn_metric"],
        )

    transform = FeatureTransform(feature_selector, parameters, seed)
    selected_source_x = transform.fit_transform(selected_source_x, selected_source_y)
    selected_target_x = transform.transform(selected_target_x)
    model = _make_classifier(classifier, parameters, seed)

    if adaptation == "cde_smote":
        counts = np.bincount(selected_source_y, minlength=2)
        minority_count = int(np.min(counts))
        if minority_count >= 2 and counts[0] != counts[1]:
            sampler = SMOTE(
                k_neighbors=min(parameters["cde_neighbors"], minority_count - 1),
                random_state=seed,
            )
            selected_source_x, selected_source_y = sampler.fit_resample(
                selected_source_x, selected_source_y
            )
    if classifier == "knn":
        model.n_neighbors = min(model.n_neighbors, len(selected_source_x))
    model.fit(selected_source_x, selected_source_y)
    return model, transform, selected_target_x


def _pareto_front(candidates: Sequence[Candidate]) -> List[Candidate]:
    front = []
    for candidate in candidates:
        dominated = any(
            (other.auc >= candidate.auc and other.f1 >= candidate.f1)
            and (other.auc > candidate.auc or other.f1 > candidate.f1)
            for other in candidates
        )
        if not dominated:
            front.append(candidate)
    return sorted(front, key=lambda item: (item.auc + item.f1, item.auc), reverse=True)


def _evaluate_candidate(
    candidate: Candidate,
    sources: Sequence[LabeledProject],
    seed: int,
) -> Tuple[float, float]:
    fold_count = min(3, len(sources))
    rng = np.random.default_rng(seed)
    validation_indices = rng.choice(len(sources), size=fold_count, replace=False)
    auc_values = []
    f1_values = []
    for fold_offset, validation_index in enumerate(validation_indices):
        validation = sources[int(validation_index)]
        training = [
            source for index, source in enumerate(sources) if index != int(validation_index)
        ]
        training_x = np.vstack([source.features.values for source in training])
        training_y = np.concatenate([source.labels for source in training])
        model, _, validation_x = _fit_pipeline(
            candidate.adaptation,
            candidate.feature_selector,
            candidate.classifier,
            candidate.parameters,
            training_x,
            training_y,
            validation.features.values,
            seed + fold_offset,
        )
        scores = _scores(model, validation_x)
        auc_values.append(roc_auc_score(validation.labels, scores))
        f1_values.append(f1_score(validation.labels, scores >= 0.5, zero_division=0))
    return float(np.mean(auc_values)), float(np.mean(f1_values))


def _predict_candidate(
    candidate: Candidate,
    source_x: np.ndarray,
    source_y: np.ndarray,
    target_x: np.ndarray,
    seed: int,
) -> np.ndarray:
    model, _, transformed_target = _fit_pipeline(
        candidate.adaptation,
        candidate.feature_selector,
        candidate.classifier,
        copy.deepcopy(candidate.parameters),
        source_x,
        source_y,
        target_x,
        seed,
    )
    return _scores(model, transformed_target)


def _evaluate_candidate_before_deadline(
    candidate: Candidate,
    sources: Sequence[LabeledProject],
    seed: int,
    candidate_timeout_seconds: float,
    search_deadline: float,
    allow_overtime_for_first_candidate: bool,
) -> Tuple[float, float]:
    remaining_seconds = search_deadline - time.monotonic()
    if remaining_seconds <= 0 and not allow_overtime_for_first_candidate:
        raise HardTimeoutError("Search wall-clock budget is exhausted")
    effective_timeout = (
        candidate_timeout_seconds
        if remaining_seconds <= 0
        else min(candidate_timeout_seconds, remaining_seconds)
    )
    return _call_with_hard_timeout(
        _evaluate_candidate,
        (candidate, sources, seed),
        effective_timeout,
    )


class MBLCPDPSVMethod(BaseMethod):
    name = "mbl_cpdp_sv"
    provenance = MethodProvenance(
        display_name="MBL-CPDP-SV",
        implementation="Source-validated unlabeled-target adaptation of the official bilevel design",
        fidelity="conceptual-unlabeled-adaptation",
        source_url="https://github.com/JiaxinChen-Archive/MBL-CPDP",
        source_revision="ae49526 (GitHub archive snapshot)",
        target_label_policy="Candidate pipelines are evaluated on held-out source projects only",
        notes=(
            "The official Time20.py passes Ytarget into every lower-level objective, so original "
            "MBL-CPDP cannot be run under a zero-target-label protocol unchanged. This SV variant "
            "retains upper pipeline choice, lower hyperparameter search, Pareto selection, ensemble "
            "construction, and official parameter ranges. Its score-producing safe search space "
            "contains CDE-SMOTE/NNfilter, five feature-selection choices, and fourteen classifiers."
        ),
    )

    def fit_predict(
        self,
        sources: Sequence[LabeledProject],
        target: UnlabeledProject,
        seed: int,
    ) -> PredictionResult:
        if len(sources) < 2:
            raise ValueError("MBL-CPDP-SV requires at least two source projects")
        rng = np.random.default_rng(seed)
        wall_clock_seconds = float(self.params.get("wall_clock_seconds", 3600))
        candidate_timeout_seconds = float(
            self.params.get("candidate_timeout_seconds", wall_clock_seconds)
        )
        final_fit_timeout_seconds = float(
            self.params.get("final_fit_timeout_seconds", candidate_timeout_seconds * 3)
        )
        upper_evaluations = int(self.params.get("upper_evaluations", 10000))
        lower_evaluations = int(self.params.get("lower_evaluations", 5))
        pool_size = int(self.params.get("ensemble_pool_size", 6))
        selected_size = int(self.params.get("ensemble_selected_size", 3))
        requested_workers = int(self.params.get("parallel_workers", lower_evaluations))
        if min(
            wall_clock_seconds,
            candidate_timeout_seconds,
            final_fit_timeout_seconds,
            upper_evaluations,
            lower_evaluations,
            pool_size,
            selected_size,
        ) <= 0:
            raise ValueError("MBL-CPDP-SV search and ensemble limits must be positive")
        if requested_workers < 1:
            raise ValueError("MBL-CPDP-SV parallel_workers must be at least 1")
        parallel_workers = min(requested_workers, lower_evaluations)
        blas_threads = max(1, int(os.cpu_count() or 1) // parallel_workers)
        started = time.monotonic()
        search_deadline = started + wall_clock_seconds
        last_progress = started
        candidates: List[Candidate] = []
        candidate_evaluations = 0
        candidate_timeouts = 0
        candidate_failures = 0

        executor_context = (
            ThreadPoolExecutor(max_workers=parallel_workers)
            if parallel_workers > 1
            else nullcontext(None)
        )
        with threadpool_limits(limits=blas_threads), executor_context as executor:
            for upper_index in range(upper_evaluations):
                if time.monotonic() >= search_deadline and candidates:
                    break
                adaptation = str(rng.choice(ADAPTATIONS))
                feature_selector = str(rng.choice(FEATURE_SELECTORS))
                classifier = str(rng.choice(CLASSIFIERS))
                evaluations = []
                for lower_index in range(lower_evaluations):
                    parameters = _sample_parameters(rng, feature_selector, classifier)
                    candidate = Candidate(
                        adaptation=adaptation,
                        feature_selector=feature_selector,
                        classifier=classifier,
                        parameters=parameters,
                        auc=0.0,
                        f1=0.0,
                    )
                    candidate_seed = (
                        seed
                        + upper_index * max(lower_evaluations, 1)
                        + lower_index
                    )
                    if executor is None:
                        evaluations.append((candidate, candidate_seed, None))
                    else:
                        evaluations.append(
                            (
                                candidate,
                                candidate_seed,
                                executor.submit(
                                    _evaluate_candidate_before_deadline,
                                    candidate,
                                    sources,
                                    candidate_seed,
                                    candidate_timeout_seconds,
                                    search_deadline,
                                    not candidates,
                                ),
                            )
                        )

                best_candidate = None
                for candidate, candidate_seed, future in evaluations:
                    if (
                        future is None
                        and time.monotonic() >= search_deadline
                        and (candidates or best_candidate is not None)
                    ):
                        break
                    candidate_evaluations += 1
                    try:
                        if future is None:
                            candidate.auc, candidate.f1 = (
                                _evaluate_candidate_before_deadline(
                                    candidate,
                                    sources,
                                    candidate_seed,
                                    candidate_timeout_seconds,
                                    search_deadline,
                                    not candidates and best_candidate is None,
                                )
                            )
                        else:
                            candidate.auc, candidate.f1 = future.result()
                    except HardTimeoutError:
                        candidate_timeouts += 1
                        print(
                            f"TIMEOUT mbl_cpdp_sv stage=search "
                            f"classifier={candidate.classifier} "
                            f"selector={candidate.feature_selector} "
                            f"limit={candidate_timeout_seconds:.0f}s",
                            flush=True,
                        )
                        continue
                    except (
                        ArithmeticError,
                        RuntimeError,
                        ValueError,
                        np.linalg.LinAlgError,
                    ):
                        candidate_failures += 1
                        continue
                    if best_candidate is None or (
                        candidate.auc + candidate.f1
                        > best_candidate.auc + best_candidate.f1
                    ):
                        best_candidate = candidate
                if best_candidate is not None:
                    candidates.append(best_candidate)

                now = time.monotonic()
                if now - last_progress >= 60.0:
                    print(
                        f"PROGRESS mbl_cpdp_sv upper_candidates={len(candidates)} "
                        f"elapsed={now - started:.0f}s",
                        flush=True,
                    )
                    last_progress = now

        search_finished = time.monotonic()

        if not candidates:
            raise RuntimeError("MBL-CPDP-SV search produced no valid candidate pipeline")
        front = _pareto_front(candidates)
        pool = front[: min(pool_size, len(front))]
        requested_ensemble_size = min(selected_size, len(pool))

        source_x = np.vstack([source.features.values for source in sources])
        source_y = np.concatenate([source.labels for source in sources])
        predictions = []
        selected = []
        final_fit_attempts = 0
        final_fit_timeouts = 0
        final_fit_failures = 0
        for candidate_index, candidate in enumerate(pool):
            if len(predictions) >= requested_ensemble_size:
                break
            final_fit_attempts += 1
            try:
                prediction = _call_with_hard_timeout(
                    _predict_candidate,
                    (
                        candidate,
                        source_x,
                        source_y,
                        target.features.values,
                        seed + 100000 + candidate_index,
                    ),
                    final_fit_timeout_seconds,
                )
            except HardTimeoutError:
                final_fit_timeouts += 1
                print(
                    f"TIMEOUT mbl_cpdp_sv stage=final_fit "
                    f"classifier={candidate.classifier} "
                    f"selector={candidate.feature_selector} "
                    f"limit={final_fit_timeout_seconds:.0f}s",
                    flush=True,
                )
                continue
            except (ArithmeticError, RuntimeError, ValueError, np.linalg.LinAlgError):
                final_fit_failures += 1
                continue
            predictions.append(np.asarray(prediction, dtype=np.float64))
            selected.append(candidate)
        if not predictions:
            raise RuntimeError("MBL-CPDP-SV final ensemble produced no valid pipeline")
        scores = np.mean(np.vstack(predictions), axis=0)
        finished = time.monotonic()

        return PredictionResult(
            scores=scores,
            threshold=0.5,
            metadata={
                "variant": "MBL-CPDP-SV",
                "target_labels_used": False,
                "evaluated_upper_candidates": len(candidates),
                "pareto_front_size": len(front),
                "ensemble_pool_size": len(pool),
                "ensemble_size": len(selected),
                "parallel_workers": parallel_workers,
                "blas_threads_per_worker": blas_threads,
                "wall_clock_seconds": wall_clock_seconds,
                "candidate_timeout_seconds": candidate_timeout_seconds,
                "final_fit_timeout_seconds": final_fit_timeout_seconds,
                "candidate_evaluations": candidate_evaluations,
                "candidate_timeouts": candidate_timeouts,
                "candidate_failures": candidate_failures,
                "final_fit_attempts": final_fit_attempts,
                "final_fit_timeouts": final_fit_timeouts,
                "final_fit_failures": final_fit_failures,
                "search_seconds": float(search_finished - started),
                "total_method_seconds": float(finished - started),
                "selected_pipelines": [
                    {
                        "adaptation": candidate.adaptation,
                        "feature_selector": candidate.feature_selector,
                        "classifier": candidate.classifier,
                        "source_validation_auc": candidate.auc,
                        "source_validation_f1": candidate.f1,
                        "parameters": candidate.parameters,
                    }
                    for candidate in selected
                ],
                "imbalance_handling": "CDE-SMOTE when selected by upper-level search",
            },
        )
