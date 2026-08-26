from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch


TensorTuple = Tuple[torch.Tensor, ...]


def resolve_torch_device(requested: str) -> torch.device:
    normalized = str(requested).strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("FedCPDP requested CUDA, but torch.cuda.is_available() is false")
    if normalized not in {"cpu", "cuda"}:
        raise ValueError("FedCPDP device must be 'auto', 'cpu', or 'cuda'")
    return torch.device(normalized)


def resolve_torch_dtype(requested: str) -> torch.dtype:
    normalized = str(requested).strip().lower()
    if normalized in {"float32", "fp32"}:
        return torch.float32
    if normalized in {"float64", "fp64"}:
        return torch.float64
    raise ValueError("FedCPDP dtype must be 'float32' or 'float64'")


@dataclass
class _AdamState:
    first_moments: List[torch.Tensor]
    second_moments: List[torch.Tensor]
    step: int = 0

    @classmethod
    def create(cls, parameters: TensorTuple) -> "_AdamState":
        return cls(
            first_moments=[torch.zeros_like(value) for value in parameters],
            second_moments=[torch.zeros_like(value) for value in parameters],
        )


class TorchFedCPDPCore:
    """Manual Torch port of the official NumPy FedCPDP training equations."""

    def __init__(self, official: Any, device: str, dtype: str) -> None:
        self.official = official
        self.device = resolve_torch_device(device)
        self.dtype = resolve_torch_dtype(dtype)
        self._previous_matmul_precision: str | None = None
        if self.device.type == "cuda":
            self._previous_matmul_precision = torch.get_float32_matmul_precision()
            torch.set_float32_matmul_precision("high")

    def prepare_clients(
        self, clients: Sequence[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], ...]:
        return tuple(
            {
                **client,
                "X": torch.as_tensor(
                    client["X"], dtype=self.dtype, device=self.device
                ),
                "y": torch.as_tensor(
                    client["y"], dtype=self.dtype, device=self.device
                ),
            }
            for client in clients
        )

    def prepare_values(self, values: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(values, dtype=self.dtype, device=self.device)

    def initialize_parameters(self, seed: int) -> TensorTuple:
        model = self.official.MLP2(
            self.official.HASH_DIM,
            self.official.H1_DIM,
            self.official.H2_DIM,
            1,
            l2=self.official.L2_WEIGHT_DECAY,
            seed=int(seed),
        )
        return tuple(
            torch.as_tensor(value, dtype=self.dtype, device=self.device).clone()
            for value in model.get_params()
        )

    @staticmethod
    def clone_parameters(parameters: TensorTuple) -> TensorTuple:
        return tuple(value.clone() for value in parameters)

    def initialize_ensemble_parameters(self, seeds: Sequence[int]) -> TensorTuple:
        initialized = [self.initialize_parameters(seed) for seed in seeds]
        return tuple(
            torch.stack([parameters[layer] for parameters in initialized], dim=0)
            for layer in range(len(initialized[0]))
        )

    def _generator(self, seed: int) -> torch.Generator:
        generator = torch.Generator(device=self.device.type)
        generator.manual_seed(int(seed))
        return generator

    def _forward(
        self,
        parameters: TensorTuple,
        values: torch.Tensor,
        *,
        train: bool,
        dropout_probability: float,
        generator: torch.Generator | None,
        dropout_masks: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, ...]:
        weight_1, bias_1, weight_2, bias_2, weight_3, bias_3 = parameters
        linear_1 = values @ weight_1 + bias_1
        hidden_1 = torch.relu(linear_1)
        mask_1 = None
        hidden_1_dropout = hidden_1
        if dropout_masks is not None:
            mask_1 = dropout_masks[0]
            hidden_1_dropout = hidden_1 * mask_1
        elif train and dropout_probability > 0.0:
            mask_1 = (
                torch.rand(
                    hidden_1.shape,
                    dtype=self.dtype,
                    device=self.device,
                    generator=generator,
                )
                > dropout_probability
            ).to(self.dtype) / (1.0 - dropout_probability)
            hidden_1_dropout = hidden_1 * mask_1

        linear_2 = hidden_1_dropout @ weight_2 + bias_2
        hidden_2 = torch.relu(linear_2)
        mask_2 = None
        hidden_2_dropout = hidden_2
        if dropout_masks is not None:
            mask_2 = dropout_masks[1]
            hidden_2_dropout = hidden_2 * mask_2
        elif train and dropout_probability > 0.0:
            mask_2 = (
                torch.rand(
                    hidden_2.shape,
                    dtype=self.dtype,
                    device=self.device,
                    generator=generator,
                )
                > dropout_probability
            ).to(self.dtype) / (1.0 - dropout_probability)
            hidden_2_dropout = hidden_2 * mask_2

        logits = hidden_2_dropout @ weight_3 + bias_3
        probabilities = 1.0 / (1.0 + torch.exp(-logits))
        return (
            linear_1,
            hidden_1,
            mask_1,
            linear_2,
            hidden_2,
            mask_2,
            logits,
            probabilities,
        )

    def loss_and_grads(
        self,
        parameters: TensorTuple,
        values: torch.Tensor,
        labels: torch.Tensor,
        *,
        global_parameters: TensorTuple | None,
        fedprox_mu: float,
        align_mean_weight: float,
        align_variance_weight: float,
        global_mean: torch.Tensor | None,
        global_variance: torch.Tensor | None,
        use_focal: bool,
        focal_gamma: float,
        dropout_probability: float,
        generator: torch.Generator | None,
        dropout_masks: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, TensorTuple, torch.Tensor, torch.Tensor]:
        sample_count = values.shape[0]
        (
            linear_1,
            hidden_1,
            mask_1,
            linear_2,
            hidden_2,
            mask_2,
            _,
            probabilities,
        ) = self._forward(
            parameters,
            values,
            train=True,
            dropout_probability=dropout_probability,
            generator=generator,
            dropout_masks=dropout_masks,
        )
        labels = labels.reshape(-1, 1).to(dtype=self.dtype)
        epsilon = 1e-12
        positive = torch.clamp(labels.sum(), min=1.0)
        negative = torch.clamp(sample_count - labels.sum(), min=1.0)
        positive_weight = negative / positive

        if use_focal:
            focal_positive = (
                -positive_weight
                * ((1.0 - probabilities) ** focal_gamma)
                * labels
                * torch.log(probabilities + epsilon)
            )
            focal_negative = (
                -(probabilities**focal_gamma)
                * (1.0 - labels)
                * torch.log(1.0 - probabilities + epsilon)
            )
            classification_loss = (focal_positive + focal_negative).mean()
            logits_gradient = (
                (probabilities - labels)
                * (1.0 + (positive_weight - 1.0) * labels)
                * (
                    (1.0 - probabilities) ** focal_gamma
                    + probabilities**focal_gamma
                )
                / sample_count
            )
        else:
            classification_loss = -(
                positive_weight * labels * torch.log(probabilities + epsilon)
                + (1.0 - labels) * torch.log(1.0 - probabilities + epsilon)
            ).mean()
            logits_gradient = (
                (probabilities - labels)
                * (1.0 + (positive_weight - 1.0) * labels)
                / sample_count
            )

        weight_1, bias_1, weight_2, bias_2, weight_3, bias_3 = parameters
        l2_loss = 0.5 * self.official.L2_WEIGHT_DECAY * (
            torch.sum(weight_1**2)
            + torch.sum(weight_2**2)
            + torch.sum(weight_3**2)
        )
        prox_loss = torch.zeros((), dtype=self.dtype, device=self.device)
        if fedprox_mu > 0.0 and global_parameters is not None:
            prox_loss = 0.5 * fedprox_mu * sum(
                torch.sum((parameter - reference) ** 2)
                for parameter, reference in zip(parameters, global_parameters)
            )

        hidden_mean = hidden_2.mean(dim=0)
        hidden_variance = hidden_2.var(dim=0, unbiased=False)
        alignment_loss = torch.zeros((), dtype=self.dtype, device=self.device)
        if align_mean_weight > 0.0 and global_mean is not None:
            alignment_loss = alignment_loss + 0.5 * align_mean_weight * torch.sum(
                (hidden_mean - global_mean) ** 2
            )
        if align_variance_weight > 0.0 and global_variance is not None:
            alignment_loss = alignment_loss + 0.5 * align_variance_weight * torch.sum(
                (hidden_variance - global_variance) ** 2
            )
        loss = classification_loss + l2_loss + prox_loss + alignment_loss

        hidden_2_dropout = hidden_2 if mask_2 is None else hidden_2 * mask_2
        grad_weight_3 = (
            hidden_2_dropout.T @ logits_gradient
            + self.official.L2_WEIGHT_DECAY * weight_3
        )
        grad_bias_3 = logits_gradient.sum(dim=0)
        grad_hidden_2 = logits_gradient @ weight_3.T
        if mask_2 is not None:
            grad_hidden_2 = grad_hidden_2 * mask_2
        grad_linear_2 = grad_hidden_2 * (linear_2 > 0.0)

        hidden_1_dropout = hidden_1 if mask_1 is None else hidden_1 * mask_1
        grad_weight_2 = (
            hidden_1_dropout.T @ grad_linear_2
            + self.official.L2_WEIGHT_DECAY * weight_2
        )
        grad_bias_2 = grad_linear_2.sum(dim=0)
        grad_hidden_1 = grad_linear_2 @ weight_2.T
        if mask_1 is not None:
            grad_hidden_1 = grad_hidden_1 * mask_1
        grad_linear_1 = grad_hidden_1 * (linear_1 > 0.0)
        grad_weight_1 = (
            values.T @ grad_linear_1 + self.official.L2_WEIGHT_DECAY * weight_1
        )
        grad_bias_1 = grad_linear_1.sum(dim=0)

        if fedprox_mu > 0.0 and global_parameters is not None:
            gradients = [
                grad_weight_1,
                grad_bias_1,
                grad_weight_2,
                grad_bias_2,
                grad_weight_3,
                grad_bias_3,
            ]
            gradients = [
                gradient + fedprox_mu * (parameter - reference)
                for gradient, parameter, reference in zip(
                    gradients, parameters, global_parameters
                )
            ]
            (
                grad_weight_1,
                grad_bias_1,
                grad_weight_2,
                grad_bias_2,
                grad_weight_3,
                grad_bias_3,
            ) = gradients

        if align_mean_weight > 0.0 and global_mean is not None:
            grad_hidden_mean = (
                (hidden_mean - global_mean) * align_mean_weight / sample_count
            )
            grad_linear_mean = torch.where(
                linear_2 > 0.0,
                grad_hidden_mean.unsqueeze(0),
                torch.zeros((), dtype=self.dtype, device=self.device),
            )
            grad_weight_2 = grad_weight_2 + hidden_1_dropout.T @ grad_linear_mean
            grad_bias_2 = grad_bias_2 + grad_linear_mean.sum(dim=0)

        if align_variance_weight > 0.0 and global_variance is not None:
            grad_variance = (
                hidden_variance - global_variance
            ) * align_variance_weight
            grad_hidden_variance = (
                2.0 / sample_count
            ) * (hidden_2 - hidden_mean) * grad_variance
            grad_linear_variance = torch.where(
                linear_2 > 0.0,
                grad_hidden_variance,
                torch.zeros((), dtype=self.dtype, device=self.device),
            )
            grad_weight_2 = (
                grad_weight_2 + hidden_1_dropout.T @ grad_linear_variance
            )
            grad_bias_2 = grad_bias_2 + grad_linear_variance.sum(dim=0)

        gradients = (
            grad_weight_1,
            grad_bias_1,
            grad_weight_2,
            grad_bias_2,
            grad_weight_3,
            grad_bias_3,
        )
        return loss, gradients, hidden_mean, hidden_variance

    def _loss_and_grads_with_masks(
        self,
        parameters: TensorTuple,
        values: torch.Tensor,
        labels: torch.Tensor,
        global_parameters: TensorTuple,
        global_mean: torch.Tensor,
        global_variance: torch.Tensor,
        mask_1: torch.Tensor,
        mask_2: torch.Tensor,
    ) -> Tuple[torch.Tensor, TensorTuple, torch.Tensor, torch.Tensor]:
        return self.loss_and_grads(
            parameters,
            values,
            labels,
            global_parameters=global_parameters,
            fedprox_mu=(
                self.official.FEDPROX_MU if self.official.USE_FEDPROX else 0.0
            ),
            align_mean_weight=(
                self.official.ALIGN_MEAN_WEIGHT if self.official.USE_ALIGN else 0.0
            ),
            align_variance_weight=(
                self.official.ALIGN_VAR_WEIGHT if self.official.USE_ALIGN else 0.0
            ),
            global_mean=global_mean,
            global_variance=global_variance,
            use_focal=self.official.USE_FOCAL,
            focal_gamma=self.official.FOCAL_GAMMA,
            dropout_probability=self.official.DROPOUT_P,
            generator=None,
            dropout_masks=(mask_1, mask_2),
        )

    def _adam_step(
        self,
        parameters: TensorTuple,
        gradients: TensorTuple,
        state: _AdamState,
        learning_rate: float,
    ) -> TensorTuple:
        state.step += 1
        beta_1 = self.official.ADAM_BETA1
        beta_2 = self.official.ADAM_BETA2
        epsilon = self.official.ADAM_EPS
        updated = []
        for index, (parameter, gradient) in enumerate(zip(parameters, gradients)):
            first_moment = state.first_moments[index]
            second_moment = state.second_moments[index]
            first_moment.mul_(beta_1).add_(gradient, alpha=1.0 - beta_1)
            second_moment.mul_(beta_2).addcmul_(
                gradient, gradient, value=1.0 - beta_2
            )
            corrected_first = first_moment / (1.0 - beta_1**state.step)
            corrected_second = second_moment / (1.0 - beta_2**state.step)
            updated.append(
                parameter
                - learning_rate
                * corrected_first
                / (torch.sqrt(corrected_second) + epsilon)
            )
        return tuple(updated)

    def local_train(
        self,
        client: Dict[str, Any],
        global_parameters: TensorTuple,
        global_mean: torch.Tensor,
        global_variance: torch.Tensor,
        seed: int,
    ) -> Tuple[TensorTuple, torch.Tensor, torch.Tensor, torch.Tensor]:
        values = client["X"]
        labels = client["y"]
        sample_count = len(labels)
        generator = self._generator(seed)
        parameters = self.clone_parameters(global_parameters)
        optimizer = _AdamState.create(parameters)
        fedprox_mu = self.official.FEDPROX_MU if self.official.USE_FEDPROX else 0.0
        batches_per_epoch = int(np.ceil(sample_count / self.official.BATCH_SIZE))
        total_steps = batches_per_epoch * self.official.LOCAL_EPOCHS
        step = 0
        losses = []
        hidden_means = []
        hidden_variances = []

        for _ in range(self.official.LOCAL_EPOCHS):
            indices = torch.randperm(
                sample_count, generator=generator, device=self.device
            )
            for start in range(0, sample_count, self.official.BATCH_SIZE):
                batch_indices = indices[start : start + self.official.BATCH_SIZE]
                batch_values = values[batch_indices]
                batch_labels = labels[batch_indices]
                if self.official.USE_COSINE_LR:
                    progress = step / max(1, total_steps)
                    learning_rate = 0.5 * self.official.LEARNING_RATE * (
                        1.0 + np.cos(np.pi * progress)
                    )
                else:
                    learning_rate = self.official.LEARNING_RATE
                loss, gradients, hidden_mean, hidden_variance = self.loss_and_grads(
                    parameters,
                    batch_values,
                    batch_labels,
                    global_parameters=global_parameters,
                    fedprox_mu=fedprox_mu,
                    align_mean_weight=(
                        self.official.ALIGN_MEAN_WEIGHT
                        if self.official.USE_ALIGN
                        else 0.0
                    ),
                    align_variance_weight=(
                        self.official.ALIGN_VAR_WEIGHT
                        if self.official.USE_ALIGN
                        else 0.0
                    ),
                    global_mean=global_mean,
                    global_variance=global_variance,
                    use_focal=self.official.USE_FOCAL,
                    focal_gamma=self.official.FOCAL_GAMMA,
                    dropout_probability=self.official.DROPOUT_P,
                    generator=generator,
                )
                parameters = self._adam_step(
                    parameters, gradients, optimizer, float(learning_rate)
                )
                losses.append(loss)
                hidden_means.append(hidden_mean)
                hidden_variances.append(hidden_variance)
                step += 1

        delta = tuple(
            parameter - global_parameter
            for parameter, global_parameter in zip(parameters, global_parameters)
        )
        return (
            delta,
            torch.stack(losses).mean(),
            torch.stack(hidden_means).mean(dim=0),
            torch.stack(hidden_variances).mean(dim=0),
        )

    def _dropout_masks_ensemble(
        self,
        batch_size: int,
        generators: Sequence[torch.Generator],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        probability = float(self.official.DROPOUT_P)
        if probability <= 0.0:
            return (
                torch.ones(
                    (len(generators), batch_size, self.official.H1_DIM),
                    dtype=self.dtype,
                    device=self.device,
                ),
                torch.ones(
                    (len(generators), batch_size, self.official.H2_DIM),
                    dtype=self.dtype,
                    device=self.device,
                ),
            )

        scale = 1.0 / (1.0 - probability)
        first_masks = []
        second_masks = []
        for generator in generators:
            first_masks.append(
                (
                    torch.rand(
                        (batch_size, self.official.H1_DIM),
                        dtype=self.dtype,
                        device=self.device,
                        generator=generator,
                    )
                    > probability
                ).to(self.dtype)
                * scale
            )
            second_masks.append(
                (
                    torch.rand(
                        (batch_size, self.official.H2_DIM),
                        dtype=self.dtype,
                        device=self.device,
                        generator=generator,
                    )
                    > probability
                ).to(self.dtype)
                * scale
            )
        return torch.stack(first_masks, dim=0), torch.stack(second_masks, dim=0)

    def local_train_ensemble(
        self,
        client: Dict[str, Any],
        global_parameters: TensorTuple,
        global_mean: torch.Tensor,
        global_variance: torch.Tensor,
        seeds: Sequence[int],
    ) -> Tuple[TensorTuple, torch.Tensor, torch.Tensor, torch.Tensor]:
        values = client["X"]
        labels = client["y"]
        sample_count = len(labels)
        generators = [self._generator(seed) for seed in seeds]
        parameters = self.clone_parameters(global_parameters)
        optimizer = _AdamState.create(parameters)
        batches_per_epoch = int(np.ceil(sample_count / self.official.BATCH_SIZE))
        total_steps = batches_per_epoch * self.official.LOCAL_EPOCHS
        step = 0
        losses = []
        hidden_means = []
        hidden_variances = []

        for _ in range(self.official.LOCAL_EPOCHS):
            indices = torch.stack(
                [
                    torch.randperm(
                        sample_count, generator=generator, device=self.device
                    )
                    for generator in generators
                ],
                dim=0,
            )
            for start in range(0, sample_count, self.official.BATCH_SIZE):
                batch_indices = indices[:, start : start + self.official.BATCH_SIZE]
                batch_values = values[batch_indices]
                batch_labels = labels[batch_indices]
                mask_1, mask_2 = self._dropout_masks_ensemble(
                    batch_indices.shape[1], generators
                )
                if self.official.USE_COSINE_LR:
                    progress = step / max(1, total_steps)
                    learning_rate = 0.5 * self.official.LEARNING_RATE * (
                        1.0 + np.cos(np.pi * progress)
                    )
                else:
                    learning_rate = self.official.LEARNING_RATE

                loss, gradients, hidden_mean, hidden_variance = torch.vmap(
                    self._loss_and_grads_with_masks
                )(
                    parameters,
                    batch_values,
                    batch_labels,
                    global_parameters,
                    global_mean,
                    global_variance,
                    mask_1,
                    mask_2,
                )
                parameters = self._adam_step(
                    parameters, gradients, optimizer, float(learning_rate)
                )
                losses.append(loss)
                hidden_means.append(hidden_mean)
                hidden_variances.append(hidden_variance)
                step += 1

        delta = tuple(
            parameter - global_parameter
            for parameter, global_parameter in zip(parameters, global_parameters)
        )
        return (
            delta,
            torch.stack(losses, dim=0).mean(dim=0),
            torch.stack(hidden_means, dim=0).mean(dim=0),
            torch.stack(hidden_variances, dim=0).mean(dim=0),
        )

    @staticmethod
    def _cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        epsilon = 1e-12
        return torch.dot(left, right) / (
            (torch.linalg.vector_norm(left) + epsilon)
            * (torch.linalg.vector_norm(right) + epsilon)
        )

    def _clip_and_add_noise(
        self,
        delta: TensorTuple,
        generator: torch.Generator,
    ) -> TensorTuple:
        flat = torch.cat([value.reshape(-1) for value in delta])
        norm = torch.linalg.vector_norm(flat) + 1e-12
        scale = torch.clamp(self.official.DP_CLIP_NORM / norm, max=1.0)
        flat = flat * scale
        flat = flat + torch.randn(
            flat.shape,
            dtype=self.dtype,
            device=self.device,
            generator=generator,
        ) * (self.official.DP_NOISE_MULTIPLIER * self.official.DP_CLIP_NORM)
        output = []
        offset = 0
        for value in delta:
            size = value.numel()
            output.append(flat[offset : offset + size].reshape(value.shape))
            offset += size
        return tuple(output)

    def _clip_and_add_noise_ensemble(
        self,
        delta: TensorTuple,
        generators: Sequence[torch.Generator],
    ) -> TensorTuple:
        ensemble_size = len(generators)
        flat = torch.cat(
            [value.reshape(ensemble_size, -1) for value in delta], dim=1
        )
        norm = torch.linalg.vector_norm(flat, dim=1, keepdim=True) + 1e-12
        scale = torch.clamp(self.official.DP_CLIP_NORM / norm, max=1.0)
        flat = flat * scale
        noise_scale = (
            self.official.DP_NOISE_MULTIPLIER * self.official.DP_CLIP_NORM
        )
        noise = torch.stack(
            [
                torch.randn(
                    flat.shape[1],
                    dtype=self.dtype,
                    device=self.device,
                    generator=generator,
                )
                for generator in generators
            ],
            dim=0,
        )
        flat = flat + noise * noise_scale

        output = []
        offset = 0
        for value in delta:
            size = value[0].numel()
            output.append(flat[:, offset : offset + size].reshape(value.shape))
            offset += size
        return tuple(output)

    def aggregate(
        self,
        deltas: Sequence[TensorTuple],
        losses: Sequence[torch.Tensor],
        hidden_means: Sequence[torch.Tensor],
        hidden_variances: Sequence[torch.Tensor],
        sample_counts: Sequence[int],
        server_state: Dict[str, torch.Tensor],
    ) -> TensorTuple:
        weights = torch.as_tensor(
            sample_counts, dtype=self.dtype, device=self.device
        )
        weights = weights / weights.sum().clamp_min(1e-12)
        if self.official.AGGREGATOR in {"attn", "hybrid"}:
            momentum = server_state["momentum"]
            if momentum.numel() == 0:
                momentum = torch.zeros_like(
                    torch.cat([value.reshape(-1) for value in deltas[0]])
                )
            attention_scores = []
            for delta, loss in zip(deltas, losses):
                flat_delta = torch.cat([value.reshape(-1) for value in delta])
                attention_scores.append(
                    self.official.ATTN_ALPHA * self._cosine(flat_delta, momentum)
                    + self.official.ATTN_BETA / (loss + 1e-6)
                )
            weights = weights * torch.softmax(torch.stack(attention_scores), dim=0)
            weights = weights / weights.sum().clamp_min(1e-12)

        if self.official.AGGREGATOR in {"graph", "hybrid"}:
            global_mean = server_state["global_mean"]
            global_variance = server_state["global_var"]
            graph_scores = []
            for hidden_mean, hidden_variance in zip(
                hidden_means, hidden_variances
            ):
                graph_scores.append(
                    self.official.GRAPH_GAMMA_MEAN
                    * self._cosine(hidden_mean, global_mean)
                    - self.official.GRAPH_GAMMA_VAR
                    * torch.linalg.vector_norm(hidden_variance - global_variance)
                )
            weights = weights * torch.softmax(torch.stack(graph_scores), dim=0)
            weights = weights / weights.sum().clamp_min(1e-12)

        aggregated = tuple(
            torch.sum(
                torch.stack([delta[layer] for delta in deltas], dim=0)
                * weights.reshape((-1,) + (1,) * deltas[0][layer].ndim),
                dim=0,
            )
            for layer in range(len(deltas[0]))
        )
        flat_aggregated = torch.cat([value.reshape(-1) for value in aggregated])
        if server_state["momentum"].numel() == 0:
            server_state["momentum"] = flat_aggregated.clone()
        else:
            server_state["momentum"] = (
                self.official.MOMENTUM_EMA * server_state["momentum"]
                + (1.0 - self.official.MOMENTUM_EMA) * flat_aggregated
            )
        return aggregated

    def aggregate_ensemble(
        self,
        deltas: Sequence[TensorTuple],
        losses: Sequence[torch.Tensor],
        hidden_means: Sequence[torch.Tensor],
        hidden_variances: Sequence[torch.Tensor],
        sample_counts: Sequence[int],
        server_state: Dict[str, torch.Tensor],
    ) -> TensorTuple:
        ensemble_size = deltas[0][0].shape[0]
        client_count = len(deltas)
        size_weights = torch.as_tensor(
            sample_counts, dtype=self.dtype, device=self.device
        )
        size_weights = size_weights / size_weights.sum().clamp_min(1e-12)
        weights = size_weights.unsqueeze(0).expand(ensemble_size, -1).clone()

        flat_deltas = torch.cat(
            [
                torch.stack([delta[layer] for delta in deltas], dim=1).reshape(
                    ensemble_size, client_count, -1
                )
                for layer in range(len(deltas[0]))
            ],
            dim=2,
        )
        if self.official.AGGREGATOR in {"attn", "hybrid"}:
            momentum = server_state["momentum"]
            if momentum.shape[1] == 0:
                momentum = torch.zeros_like(flat_deltas[:, 0, :])
            numerator = torch.sum(flat_deltas * momentum[:, None, :], dim=2)
            denominator = (
                torch.linalg.vector_norm(flat_deltas, dim=2) + 1e-12
            ) * (torch.linalg.vector_norm(momentum, dim=1, keepdim=True) + 1e-12)
            cosine_scores = numerator / denominator
            loss_values = torch.stack(losses, dim=1)
            attention_scores = (
                self.official.ATTN_ALPHA * cosine_scores
                + self.official.ATTN_BETA / (loss_values + 1e-6)
            )
            weights = weights * torch.softmax(attention_scores, dim=1)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)

        if self.official.AGGREGATOR in {"graph", "hybrid"}:
            mean_values = torch.stack(hidden_means, dim=1)
            variance_values = torch.stack(hidden_variances, dim=1)
            global_mean = server_state["global_mean"]
            global_variance = server_state["global_var"]
            mean_numerator = torch.sum(
                mean_values * global_mean[:, None, :], dim=2
            )
            mean_denominator = (
                torch.linalg.vector_norm(mean_values, dim=2) + 1e-12
            ) * (
                torch.linalg.vector_norm(global_mean, dim=1, keepdim=True) + 1e-12
            )
            graph_scores = (
                self.official.GRAPH_GAMMA_MEAN
                * (mean_numerator / mean_denominator)
                - self.official.GRAPH_GAMMA_VAR
                * torch.linalg.vector_norm(
                    variance_values - global_variance[:, None, :], dim=2
                )
            )
            weights = weights * torch.softmax(graph_scores, dim=1)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)

        aggregated = tuple(
            torch.sum(
                torch.stack([delta[layer] for delta in deltas], dim=1)
                * weights.reshape(
                    (ensemble_size, client_count)
                    + (1,) * (deltas[0][layer].ndim - 1)
                ),
                dim=1,
            )
            for layer in range(len(deltas[0]))
        )
        flat_aggregated = torch.cat(
            [value.reshape(ensemble_size, -1) for value in aggregated], dim=1
        )
        if server_state["momentum"].shape[1] == 0:
            server_state["momentum"] = flat_aggregated.clone()
        else:
            server_state["momentum"] = (
                self.official.MOMENTUM_EMA * server_state["momentum"]
                + (1.0 - self.official.MOMENTUM_EMA) * flat_aggregated
            )
        return aggregated

    def federated_round(
        self,
        clients: Sequence[Dict[str, Any]],
        global_parameters: TensorTuple,
        server_state: Dict[str, torch.Tensor],
        seed: int,
        noise_generator: torch.Generator,
    ) -> TensorTuple:
        deltas = []
        losses = []
        hidden_means = []
        hidden_variances = []
        sample_counts = []
        for client in clients:
            delta, loss, hidden_mean, hidden_variance = self.local_train(
                client,
                global_parameters,
                server_state["global_mean"],
                server_state["global_var"],
                seed,
            )
            if self.official.USE_DP:
                delta = self._clip_and_add_noise(delta, noise_generator)
            deltas.append(delta)
            losses.append(loss)
            hidden_means.append(hidden_mean)
            hidden_variances.append(hidden_variance)
            sample_counts.append(len(client["y"]))

        aggregated = self.aggregate(
            deltas,
            losses,
            hidden_means,
            hidden_variances,
            sample_counts,
            server_state,
        )
        updated = tuple(
            old + change for old, change in zip(global_parameters, aggregated)
        )
        normalized_counts = torch.as_tensor(
            sample_counts, dtype=self.dtype, device=self.device
        )
        normalized_counts = normalized_counts / normalized_counts.sum()
        server_state["global_mean"] = torch.sum(
            torch.stack(hidden_means, dim=0) * normalized_counts[:, None], dim=0
        )
        server_state["global_var"] = torch.sum(
            torch.stack(hidden_variances, dim=0) * normalized_counts[:, None], dim=0
        )
        return updated

    def federated_round_ensemble(
        self,
        clients: Sequence[Dict[str, Any]],
        global_parameters: TensorTuple,
        server_state: Dict[str, torch.Tensor],
        seeds: Sequence[int],
        noise_generators: Sequence[torch.Generator],
    ) -> TensorTuple:
        deltas = []
        losses = []
        hidden_means = []
        hidden_variances = []
        sample_counts = []
        for client in clients:
            delta, loss, hidden_mean, hidden_variance = self.local_train_ensemble(
                client,
                global_parameters,
                server_state["global_mean"],
                server_state["global_var"],
                seeds,
            )
            if self.official.USE_DP:
                delta = self._clip_and_add_noise_ensemble(
                    delta, noise_generators
                )
            deltas.append(delta)
            losses.append(loss)
            hidden_means.append(hidden_mean)
            hidden_variances.append(hidden_variance)
            sample_counts.append(len(client["y"]))

        aggregated = self.aggregate_ensemble(
            deltas,
            losses,
            hidden_means,
            hidden_variances,
            sample_counts,
            server_state,
        )
        updated = tuple(
            old + change for old, change in zip(global_parameters, aggregated)
        )
        normalized_counts = torch.as_tensor(
            sample_counts, dtype=self.dtype, device=self.device
        )
        normalized_counts = normalized_counts / normalized_counts.sum()
        server_state["global_mean"] = torch.sum(
            torch.stack(hidden_means, dim=1)
            * normalized_counts.reshape(1, -1, 1),
            dim=1,
        )
        server_state["global_var"] = torch.sum(
            torch.stack(hidden_variances, dim=1)
            * normalized_counts.reshape(1, -1, 1),
            dim=1,
        )
        return updated

    def train_one_seed(
        self, clients: Sequence[Dict[str, Any]], seed: int
    ) -> Tuple[TensorTuple, Dict[str, torch.Tensor]]:
        parameters = self.initialize_parameters(seed)
        server_state = {
            "global_mean": torch.zeros(
                self.official.H2_DIM, dtype=self.dtype, device=self.device
            ),
            "global_var": torch.ones(
                self.official.H2_DIM, dtype=self.dtype, device=self.device
            ),
            "momentum": torch.empty(0, dtype=self.dtype, device=self.device),
        }
        noise_generator = self._generator(int(seed) + 777)
        with torch.no_grad():
            for _ in range(self.official.ROUNDS):
                parameters = self.federated_round(
                    clients, parameters, server_state, seed, noise_generator
                )
        return parameters, server_state

    def train_ensemble(
        self, clients: Sequence[Dict[str, Any]], seeds: Sequence[int]
    ) -> Tuple[TensorTuple, Dict[str, torch.Tensor]]:
        if not seeds:
            raise ValueError("FedCPDP ensemble requires at least one seed")
        ensemble_size = len(seeds)
        parameters = self.initialize_ensemble_parameters(seeds)
        server_state = {
            "global_mean": torch.zeros(
                (ensemble_size, self.official.H2_DIM),
                dtype=self.dtype,
                device=self.device,
            ),
            "global_var": torch.ones(
                (ensemble_size, self.official.H2_DIM),
                dtype=self.dtype,
                device=self.device,
            ),
            "momentum": torch.empty(
                (ensemble_size, 0), dtype=self.dtype, device=self.device
            ),
        }
        noise_generators = [self._generator(int(seed) + 777) for seed in seeds]
        with torch.no_grad():
            for _ in range(self.official.ROUNDS):
                parameters = self.federated_round_ensemble(
                    clients,
                    parameters,
                    server_state,
                    seeds,
                    noise_generators,
                )
        return parameters, server_state

    def predict_scores(
        self, parameters: TensorTuple, values: torch.Tensor
    ) -> np.ndarray:
        with torch.no_grad():
            probabilities = self._forward(
                parameters,
                values,
                train=False,
                dropout_probability=0.0,
                generator=None,
            )[-1]
        return probabilities.reshape(-1).detach().cpu().numpy()

    def predict_ensemble_scores(
        self, parameters: TensorTuple, values: torch.Tensor
    ) -> np.ndarray:
        weight_1, bias_1, weight_2, bias_2, weight_3, bias_3 = parameters
        with torch.no_grad():
            hidden_1 = torch.relu(
                torch.matmul(values.unsqueeze(0), weight_1) + bias_1[:, None, :]
            )
            hidden_2 = torch.relu(
                torch.matmul(hidden_1, weight_2) + bias_2[:, None, :]
            )
            logits = torch.matmul(hidden_2, weight_3) + bias_3[:, None, :]
            probabilities = 1.0 / (1.0 + torch.exp(-logits))
        return probabilities.squeeze(-1).detach().cpu().numpy()

    def release(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
        if self._previous_matmul_precision is not None:
            torch.set_float32_matmul_precision(self._previous_matmul_precision)
            self._previous_matmul_precision = None
