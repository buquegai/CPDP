from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


def _probability(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


@dataclass(frozen=True)
class TransformerEncoderConfig:
    model_dim: int = 64
    latent_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    feedforward_dim: int = 128
    dropout: float = 0.1
    activation: str = "gelu"

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "TransformerEncoderConfig":
        config = cls(
            model_dim=int(values.get("model_dim", 64)),
            latent_dim=int(values.get("latent_dim", 64)),
            num_heads=int(values.get("num_heads", 4)),
            num_layers=int(values.get("num_layers", 2)),
            feedforward_dim=int(values.get("feedforward_dim", 128)),
            dropout=float(values.get("dropout", 0.1)),
            activation=str(values.get("activation", "gelu")).lower(),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.model_dim < 2 or self.latent_dim < 2:
            raise ValueError("encoder model_dim and latent_dim must be at least 2")
        if self.num_heads < 1 or self.model_dim % self.num_heads != 0:
            raise ValueError("encoder.model_dim must be divisible by encoder.num_heads")
        if self.num_layers < 1 or self.feedforward_dim < 1:
            raise ValueError("encoder layer counts and feedforward_dim must be positive")
        _probability(self.dropout, "encoder.dropout")
        if self.activation not in {"relu", "gelu"}:
            raise ValueError("encoder.activation must be 'relu' or 'gelu'")


@dataclass(frozen=True)
class TransformerPredictorConfig:
    model_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    feedforward_dim: int = 128
    dropout: float = 0.1
    activation: str = "gelu"

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "TransformerPredictorConfig":
        config = cls(
            model_dim=int(values.get("model_dim", 64)),
            num_heads=int(values.get("num_heads", 4)),
            num_layers=int(values.get("num_layers", 2)),
            feedforward_dim=int(values.get("feedforward_dim", 128)),
            dropout=float(values.get("dropout", 0.1)),
            activation=str(values.get("activation", "gelu")).lower(),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.model_dim < 2:
            raise ValueError("jepa_predictor.model_dim must be at least 2")
        if self.num_heads < 1 or self.model_dim % self.num_heads != 0:
            raise ValueError(
                "jepa_predictor.model_dim must be divisible by "
                "jepa_predictor.num_heads"
            )
        if self.num_layers < 1 or self.feedforward_dim < 1:
            raise ValueError(
                "jepa_predictor layer counts and feedforward_dim must be positive"
            )
        _probability(self.dropout, "jepa_predictor.dropout")
        if self.activation not in {"relu", "gelu"}:
            raise ValueError("jepa_predictor.activation must be 'relu' or 'gelu'")


def _transformer_stack(torch, config, model_dim: int):
    nn = torch.nn
    layer = nn.TransformerEncoderLayer(
        d_model=model_dim,
        nhead=config.num_heads,
        dim_feedforward=config.feedforward_dim,
        dropout=config.dropout,
        activation=config.activation,
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(
        layer,
        num_layers=config.num_layers,
        norm=nn.LayerNorm(model_dim),
        enable_nested_tensor=False,
    )


def build_transformer_jepa_model(
    torch,
    feature_dim: int,
    encoder_config: TransformerEncoderConfig,
    predictor_config: TransformerPredictorConfig,
):
    if feature_dim < 1:
        raise ValueError("feature_dim must be positive")
    nn = torch.nn

    class FeatureTokenizer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.value_projection = nn.Linear(1, encoder_config.model_dim)
            self.feature_embedding = nn.Parameter(
                torch.empty(1, feature_dim, encoder_config.model_dim)
            )
            self.mask_embedding = nn.Embedding(2, encoder_config.model_dim)
            self.cls_token = nn.Parameter(
                torch.empty(1, 1, encoder_config.model_dim)
            )
            self.dropout = nn.Dropout(encoder_config.dropout)
            nn.init.normal_(self.feature_embedding, mean=0.0, std=0.02)
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

        def forward(self, values, mask):
            if values.ndim != 2 or values.shape[1] != feature_dim:
                raise ValueError(
                    f"Transformer encoder expected [batch, {feature_dim}], "
                    f"got {tuple(values.shape)}"
                )
            if values.shape != mask.shape:
                raise ValueError("values and mask must have identical shapes")
            feature_tokens = self.value_projection(values.unsqueeze(-1))
            feature_tokens = feature_tokens + self.feature_embedding.to(
                dtype=feature_tokens.dtype
            )
            mask_tokens = self.mask_embedding(mask.long()).to(
                dtype=feature_tokens.dtype
            )
            feature_tokens = feature_tokens + mask_tokens
            cls_tokens = self.cls_token.to(dtype=feature_tokens.dtype).expand(
                values.shape[0], -1, -1
            )
            return self.dropout(torch.cat((cls_tokens, feature_tokens), dim=1))

    class TransformerFeatureEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.tokenizer = FeatureTokenizer()
            self.transformer = _transformer_stack(
                torch, encoder_config, encoder_config.model_dim
            )
            self.latent_projection = nn.Linear(
                encoder_config.model_dim, encoder_config.latent_dim
            )
            self.latent_norm = nn.LayerNorm(encoder_config.latent_dim)

        def encode_tokens(self, values, mask):
            encoded_tokens = self.transformer(self.tokenizer(values, mask))
            latent = self.latent_norm(self.latent_projection(encoded_tokens[:, 0]))
            return latent, encoded_tokens

        def forward(self, values, mask):
            latent, _ = self.encode_tokens(values, mask)
            return latent

    class TransformerLatentPredictor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.context_projection = nn.Linear(
                encoder_config.model_dim, predictor_config.model_dim
            )
            self.mask_projection = nn.Sequential(
                nn.Linear(feature_dim, predictor_config.model_dim),
                nn.LayerNorm(predictor_config.model_dim),
                nn.GELU(),
            )
            self.prediction_token = nn.Parameter(
                torch.empty(1, 1, predictor_config.model_dim)
            )
            self.transformer = _transformer_stack(
                torch, predictor_config, predictor_config.model_dim
            )
            self.output_projection = nn.Linear(
                predictor_config.model_dim, encoder_config.latent_dim
            )
            self.output_norm = nn.LayerNorm(encoder_config.latent_dim)
            nn.init.normal_(self.prediction_token, mean=0.0, std=0.02)

        def forward(self, context_tokens, mask):
            if mask.ndim != 2 or mask.shape[1] != feature_dim:
                raise ValueError(
                    f"Transformer predictor expected mask [batch, {feature_dim}], "
                    f"got {tuple(mask.shape)}"
                )
            projected_context = self.context_projection(context_tokens)
            mask_embedding = self.mask_projection(mask.to(projected_context.dtype))
            query = self.prediction_token.to(dtype=mask_embedding.dtype).expand(
                mask.shape[0], -1, -1
            )
            query = query + mask_embedding.unsqueeze(1)
            prediction_tokens = self.transformer(
                torch.cat((query, projected_context), dim=1)
            )
            return self.output_norm(self.output_projection(prediction_tokens[:, 0]))

    class TransformerJEPAModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.context_encoder = TransformerFeatureEncoder()
            self.target_encoder = TransformerFeatureEncoder()
            self.predictor = TransformerLatentPredictor()
            self.classifier = nn.Linear(encoder_config.latent_dim, 1)
            self.target_encoder.load_state_dict(self.context_encoder.state_dict())
            self.target_encoder.requires_grad_(False)

        def context_latent(self, values, mask=None):
            if mask is None:
                mask = torch.zeros_like(values, dtype=torch.bool)
            return self.context_encoder(values, mask)

        def target_latent(self, values):
            mask = torch.zeros_like(values, dtype=torch.bool)
            return self.target_encoder(values, mask)

        def predicted_target(self, context_values, mask):
            context, context_tokens = self.context_encoder.encode_tokens(
                context_values, mask
            )
            prediction = self.predictor(context_tokens, mask)
            return context, prediction

        def classification_logits(self, values):
            return self.classifier(self.context_latent(values)).squeeze(1)

    return TransformerJEPAModel()
