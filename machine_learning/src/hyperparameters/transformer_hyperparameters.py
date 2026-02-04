"""
Transformer hyperparameters
"""

from dataclasses import dataclass

from src.hyperparameters.base_hyperparameters import BaseHyperparameters

@dataclass
class TransformerHyperparameters(BaseHyperparameters):
    num_layers: int = 5 # number of transformer layers
    dropout: float = 0.2 # dropout rate
    hidden_size: int = 512 # hidden_size of the transformer layers
    d_ff: int = 128 # size of the fc layers inside the transformer
    n_heads: int = 8 # number of attention heads
    num_head_layers: int = 2 # number of fc layers in the ModelHead
    k: int = 2 # expansion factor in the model head
    head_act: str = "relu" # activation in the ModelHead
    loss_type: str = "CE" # type of classification loss function to use
    task: str = "classification" # task to perform in the ModelHead
    gap: bool = True # weather to use Global Average Pooling or not
