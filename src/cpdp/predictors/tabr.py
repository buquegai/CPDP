from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


def _probability(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


@dataclass(frozen=True)
class TabRClassifierConfig:
    context_size: int = 32
    hidden_dim: int = 128
    hidden_multiplier: float = 2.0
    encoder_blocks: int = 1
    predictor_blocks: int = 1
    context_dropout: float = 0.1
    dropout: float = 0.1
    candidate_batch_size: int = 1024

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "TabRClassifierConfig":
        config = cls(
            context_size=int(values.get("context_size", 32)),
            hidden_dim=int(values.get("hidden_dim", 128)),
            hidden_multiplier=float(values.get("hidden_multiplier", 2.0)),
            encoder_blocks=int(values.get("encoder_blocks", 1)),
            predictor_blocks=int(values.get("predictor_blocks", 1)),
            context_dropout=float(values.get("context_dropout", 0.1)),
            dropout=float(values.get("dropout", 0.1)),
            candidate_batch_size=int(values.get("candidate_batch_size", 1024)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.context_size < 1:
            raise ValueError("classifier.context_size must be positive")
        if self.hidden_dim < 2:
            raise ValueError("classifier.hidden_dim must be at least 2")
        if self.hidden_multiplier <= 0:
            raise ValueError("classifier.hidden_multiplier must be positive")
        if self.encoder_blocks < 0 or self.predictor_blocks < 0:
            raise ValueError("classifier block counts cannot be negative")
        _probability(self.context_dropout, "classifier.context_dropout")
        _probability(self.dropout, "classifier.dropout")
        if self.candidate_batch_size < 1:
            raise ValueError("classifier.candidate_batch_size must be positive")


def build_tabr_classifier(
    torch,
    latent_dim: int,
    config: TabRClassifierConfig,
):
    """Build the TabR E-R-P classifier for binary latent-space prediction."""

    nn = torch.nn
    functional = torch.nn.functional
    block_dim = max(1, int(config.hidden_dim * config.hidden_multiplier))

    def residual_block(prenorm: bool):
        layers = []
        if prenorm:
            layers.append(nn.LayerNorm(config.hidden_dim))
        layers.extend(
            (
                nn.Linear(config.hidden_dim, block_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(block_dim, config.hidden_dim),
                nn.Dropout(config.dropout),
            )
        )
        return nn.Sequential(*layers)

    class TabRClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_projection = nn.Linear(latent_dim, config.hidden_dim)
            self.encoder_blocks = nn.ModuleList(
                residual_block(index > 0) for index in range(config.encoder_blocks)
            )
            self.key_normalization = (
                nn.LayerNorm(config.hidden_dim) if config.encoder_blocks > 0 else None
            )
            self.key_projection = nn.Linear(config.hidden_dim, config.hidden_dim)
            self.label_encoder = nn.Embedding(2, config.hidden_dim)
            self.context_transform = nn.Sequential(
                nn.Linear(config.hidden_dim, block_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(block_dim, config.hidden_dim, bias=False),
            )
            self.context_dropout = nn.Dropout(config.context_dropout)
            self.predictor_blocks = nn.ModuleList(
                residual_block(True) for _ in range(config.predictor_blocks)
            )
            self.head = nn.Sequential(
                nn.LayerNorm(config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, 1),
            )
            nn.init.uniform_(self.label_encoder.weight, -1.0, 1.0)

        def encode(self, latent):
            encoded = self.input_projection(latent)
            for block in self.encoder_blocks:
                encoded = encoded + block(encoded)
            key_input = (
                encoded
                if self.key_normalization is None
                else self.key_normalization(encoded)
            )
            return encoded, self.key_projection(key_input)

        def candidate_keys(self, candidate_latent):
            outputs = []
            for start in range(0, len(candidate_latent), config.candidate_batch_size):
                _, keys = self.encode(
                    candidate_latent[start : start + config.candidate_batch_size]
                )
                outputs.append(keys)
            return torch.cat(outputs, dim=0)

        def select_context_indices(
            self,
            query_keys,
            candidate_keys,
            query_indices=None,
        ):
            if query_keys.ndim != 2 or candidate_keys.ndim != 2:
                raise ValueError("TabR keys must be two-dimensional")
            if query_keys.shape[1] != candidate_keys.shape[1]:
                raise ValueError("TabR query and candidate key dimensions must match")
            excluded = 1 if query_indices is not None else 0
            available = len(candidate_keys) - excluded
            if available < 1:
                raise ValueError("TabR requires at least one non-self retrieval candidate")
            context_size = min(config.context_size, available)
            query_float = query_keys.float()
            candidate_float = candidate_keys.float()
            distances = (
                query_float.square().sum(dim=1, keepdim=True)
                - 2.0 * query_float @ candidate_float.T
                + candidate_float.square().sum(dim=1).unsqueeze(0)
            ).clamp_min_(0.0)
            if query_indices is not None:
                query_indices = query_indices.to(device=distances.device, dtype=torch.long)
                if query_indices.ndim != 1 or len(query_indices) != len(query_keys):
                    raise ValueError("TabR query indices must match the query batch")
                if bool(((query_indices < 0) | (query_indices >= len(candidate_keys))).any()):
                    raise ValueError("TabR query indices are outside the candidate pool")
                distances.scatter_(1, query_indices[:, None], float("inf"))
            return torch.topk(
                distances,
                k=context_size,
                dim=1,
                largest=False,
                sorted=True,
            ).indices

        def predict_from_context(
            self,
            query_encoded,
            query_keys,
            context_latent,
            context_labels,
        ):
            batch_size, context_size, _ = context_latent.shape
            _, context_keys = self.encode(context_latent.flatten(0, 1))
            context_keys = context_keys.reshape(batch_size, context_size, -1)
            similarities = (
                -query_keys.float().square().sum(dim=1, keepdim=True)
                + 2.0
                * torch.bmm(
                    query_keys.float().unsqueeze(1),
                    context_keys.float().transpose(1, 2),
                ).squeeze(1)
                - context_keys.float().square().sum(dim=2)
            )
            probabilities = self.context_dropout(
                functional.softmax(similarities, dim=1)
            ).to(query_encoded.dtype)
            label_embeddings = self.label_encoder(context_labels.long())
            values = label_embeddings + self.context_transform(
                query_keys[:, None] - context_keys
            )
            context = torch.bmm(probabilities[:, None], values).squeeze(1)
            encoded = query_encoded + context
            for block in self.predictor_blocks:
                encoded = encoded + block(encoded)
            return self.head(encoded).squeeze(1)

    return TabRClassifier()
