from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Tuple

from cpdp.predictors.tabr import TabRClassifierConfig, build_tabr_classifier


def _positive_int_tuple(values, name: str) -> Tuple[int, ...]:
    parsed = tuple(int(value) for value in values)
    if not parsed or min(parsed) < 1:
        raise ValueError(f"{name} must contain positive dimensions")
    return parsed


@dataclass(frozen=True)
class KANEncoderConfig:
    hidden_dims: Tuple[int, ...] = (128, 64)
    latent_dim: int = 64
    grid_size: int = 5
    spline_order: int = 3
    grid_range: Tuple[float, float] = (-6.0, 6.0)
    base_scale: float = 1.0
    spline_scale: float = 0.1
    dropout: float = 0.1

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "KANEncoderConfig":
        grid_range = tuple(float(value) for value in values.get("grid_range", (-6.0, 6.0)))
        config = cls(
            hidden_dims=_positive_int_tuple(
                values.get("hidden_dims", (128, 64)), "encoder.hidden_dims"
            ),
            latent_dim=int(values.get("latent_dim", 64)),
            grid_size=int(values.get("grid_size", 5)),
            spline_order=int(values.get("spline_order", 3)),
            grid_range=grid_range,
            base_scale=float(values.get("base_scale", 1.0)),
            spline_scale=float(values.get("spline_scale", 0.1)),
            dropout=float(values.get("dropout", 0.1)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.latent_dim < 2:
            raise ValueError("encoder.latent_dim must be at least 2")
        if self.grid_size < 2:
            raise ValueError("encoder.grid_size must be at least 2")
        if self.spline_order < 1:
            raise ValueError("encoder.spline_order must be positive")
        if len(self.grid_range) != 2 or self.grid_range[0] >= self.grid_range[1]:
            raise ValueError("encoder.grid_range must contain increasing bounds")
        if self.base_scale <= 0 or self.spline_scale <= 0:
            raise ValueError("encoder KAN scales must be positive")
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError("encoder.dropout must be between 0 and 1")


def build_kan_linear(
    torch,
    input_dim: int,
    output_dim: int,
    config: KANEncoderConfig,
):
    nn = torch.nn
    functional = torch.nn.functional
    grid_step = (config.grid_range[1] - config.grid_range[0]) / config.grid_size
    knots = (
        torch.arange(
            -config.spline_order,
            config.grid_size + config.spline_order + 1,
            dtype=torch.float32,
        )
        * grid_step
        + config.grid_range[0]
    )

    class KANLinear(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.base_weight = nn.Parameter(torch.empty(output_dim, input_dim))
            self.base_bias = nn.Parameter(torch.empty(output_dim))
            self.spline_weight = nn.Parameter(
                torch.empty(
                    output_dim,
                    input_dim,
                    config.grid_size + config.spline_order,
                )
            )
            self.spline_scaler = nn.Parameter(torch.empty(output_dim, input_dim))
            self.register_buffer("grid", knots.expand(input_dim, -1).contiguous())
            self.reset_parameters()

        def reset_parameters(self) -> None:
            nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
            self.base_weight.data.mul_(config.base_scale)
            fan_in = max(1, self.base_weight.shape[1])
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.base_bias, -bound, bound)
            nn.init.normal_(
                self.spline_weight,
                mean=0.0,
                std=config.spline_scale / max(1, config.grid_size),
            )
            nn.init.ones_(self.spline_scaler)

        def b_splines(self, values):
            if values.ndim != 2 or values.shape[1] != self.input_dim:
                raise ValueError(
                    f"KANLinear expected [batch, {self.input_dim}], got {tuple(values.shape)}"
                )
            expanded = values.unsqueeze(-1)
            grid = self.grid.to(dtype=values.dtype)
            bases = (
                (expanded >= grid[:, :-1]) & (expanded < grid[:, 1:])
            ).to(values.dtype)
            for order in range(1, config.spline_order + 1):
                left_denominator = grid[:, order:-1] - grid[:, : -(order + 1)]
                right_denominator = grid[:, order + 1 :] - grid[:, 1:-order]
                left = (
                    (expanded - grid[:, : -(order + 1)])
                    / left_denominator.clamp_min(torch.finfo(values.dtype).eps)
                    * bases[:, :, :-1]
                )
                right = (
                    (grid[:, order + 1 :] - expanded)
                    / right_denominator.clamp_min(torch.finfo(values.dtype).eps)
                    * bases[:, :, 1:]
                )
                bases = left + right
            return bases.contiguous()

        def forward(self, values):
            base = functional.linear(
                functional.silu(values), self.base_weight, self.base_bias
            )
            bases = self.b_splines(values)
            scaled_splines = self.spline_weight * self.spline_scaler.unsqueeze(-1)
            spline = functional.linear(
                bases.reshape(values.shape[0], -1),
                scaled_splines.reshape(self.output_dim, -1),
            )
            return base + spline

    return KANLinear()


def build_kan_encoder(torch, feature_dim: int, config: KANEncoderConfig):
    nn = torch.nn
    dimensions = (feature_dim * 2,) + config.hidden_dims + (config.latent_dim,)
    layers = []
    for index, (input_dim, output_dim) in enumerate(zip(dimensions[:-1], dimensions[1:])):
        layers.append(build_kan_linear(torch, input_dim, output_dim, config))
        layers.append(nn.LayerNorm(output_dim))
        if index < len(dimensions) - 2:
            layers.extend((nn.SiLU(), nn.Dropout(config.dropout)))
    network = nn.Sequential(*layers)

    class KANEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.network = network

        def forward(self, values, mask):
            if values.shape != mask.shape:
                raise ValueError("values and mask must have identical shapes")
            return self.network(torch.cat((values, mask.to(values.dtype)), dim=1))

    return KANEncoder()


def build_kan_jepa_model(
    torch,
    feature_dim: int,
    encoder_config: KANEncoderConfig,
    predictor_hidden_dims: Tuple[int, ...],
    mask_embedding_dim: int,
    predictor_dropout: float,
    classifier_config: TabRClassifierConfig | None = None,
):
    nn = torch.nn
    classifier_config = classifier_config or TabRClassifierConfig()

    def predictor_network():
        dimensions = (
            encoder_config.latent_dim + mask_embedding_dim,
        ) + predictor_hidden_dims + (encoder_config.latent_dim,)
        layers = []
        for index, (input_dim, output_dim) in enumerate(
            zip(dimensions[:-1], dimensions[1:])
        ):
            layers.append(nn.Linear(input_dim, output_dim))
            if index < len(dimensions) - 2:
                layers.extend(
                    (nn.LayerNorm(output_dim), nn.SiLU(), nn.Dropout(predictor_dropout))
                )
        return nn.Sequential(*layers)

    class KANJEPAModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.context_encoder = build_kan_encoder(torch, feature_dim, encoder_config)
            self.target_encoder = build_kan_encoder(torch, feature_dim, encoder_config)
            self.mask_projection = nn.Sequential(
                nn.Linear(feature_dim, mask_embedding_dim),
                nn.LayerNorm(mask_embedding_dim),
                nn.SiLU(),
            )
            self.predictor = predictor_network()
            self.classifier = build_tabr_classifier(
                torch,
                latent_dim=encoder_config.latent_dim,
                config=classifier_config,
            )
            self.target_encoder.load_state_dict(self.context_encoder.state_dict())
            self.target_encoder.requires_grad_(False)

        def context_latent(self, values, mask=None):
            if mask is None:
                mask = torch.zeros_like(values)
            return self.context_encoder(values, mask)

        def target_latent(self, values):
            mask = torch.zeros_like(values)
            return self.target_encoder(values, mask)

        def predicted_target(self, context_values, mask):
            context = self.context_latent(context_values, mask)
            mask_embedding = self.mask_projection(mask.to(context_values.dtype))
            prediction = self.predictor(torch.cat((context, mask_embedding), dim=1))
            return context, prediction

        def classification_candidate_keys(self, candidate_values):
            with torch.no_grad():
                candidate_latent = torch.cat(
                    [
                        self.context_latent(
                            candidate_values[
                                start : start + classifier_config.candidate_batch_size
                            ]
                        )
                        for start in range(
                            0,
                            len(candidate_values),
                            classifier_config.candidate_batch_size,
                        )
                    ],
                    dim=0,
                )
                return self.classifier.candidate_keys(candidate_latent)

        def classification_logits(
            self,
            values,
            candidate_values,
            candidate_labels,
            query_indices=None,
            candidate_keys=None,
        ):
            query_latent = self.context_latent(values)
            query_encoded, query_keys = self.classifier.encode(query_latent)
            with torch.no_grad():
                if candidate_keys is None:
                    candidate_keys = self.classification_candidate_keys(candidate_values)
                context_indices = self.classifier.select_context_indices(
                    query_keys.detach(),
                    candidate_keys,
                    query_indices,
                )
            context_values = candidate_values[context_indices]
            context_latent = self.context_latent(
                context_values.flatten(0, 1)
            ).reshape(context_values.shape[0], context_values.shape[1], -1)
            context_labels = candidate_labels[context_indices]
            return self.classifier.predict_from_context(
                query_encoded,
                query_keys,
                context_latent,
                context_labels,
            )

    return KANJEPAModel()


def update_ema_target(torch, model, momentum: float) -> None:
    with torch.no_grad():
        for target_parameter, context_parameter in zip(
            model.target_encoder.parameters(), model.context_encoder.parameters()
        ):
            target_parameter.mul_(momentum).add_(
                context_parameter, alpha=1.0 - momentum
            )
