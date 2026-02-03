"""
Transformer hyperparameters
"""

from dataclasses import dataclass

from src.hyperparameters.base_hyperparameters import BaseHyperparameters

@dataclass
class TransformerHyperparameters(BaseHyperparameters):
    num_layers: int = 5
    dropout: float = 0.2
    hidden_size: int = 512
    d_ff: int = 128
    n_heads: int = 8
