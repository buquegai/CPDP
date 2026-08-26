"""Interchangeable prediction backends for proposed CPDP methods."""

from cpdp.predictors.base import BasePredictor
from cpdp.predictors.kan_jepa import KANJEPAPredictor
from cpdp.predictors.mlp import MLPPredictor
from cpdp.predictors.transformer_jepa import TransformerJEPAPredictor

__all__ = [
    "BasePredictor",
    "KANJEPAPredictor",
    "MLPPredictor",
    "TransformerJEPAPredictor",
]
