"""
Python module implementing the Transformer model
"""

import math
import torch
from torch import nn

from src.hyperparameter.transformer_hyperparameters import TransformerHyperparameters

# Transformer Encoder
from transformer_encoder import TransformerEncoder
from transformer_encoder.utils import PositionalEncoding, WarmupOptimizer

class DynamicPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super(DynamicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            x: `embeddings`, shape (batch, seq_len, d_model)

        Returns:
            `encoder input`, shape (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        pe = torch.zeros(seq_len, self.d_model, device=x.device)
        position = torch.arange(0, seq_len, device=x.device).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=x.device).float()
            * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)
        x = x + pe
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(
        self,
         hyperparameters: TransformerHyperparameters,
         num_features: int,
         window_size: int,
         num_out_classes: int,
         **_kwargs
    ):
        super().__init__()


        self.input_projection = nn.Linear(num_features, hyperparameters.hidden_size)


        self.dynamic_positional_encoding = DynamicPositionalEncoding(
            d_model=hyperparameters.hidden_size,
            dropout=hyperparameters.dropout
        )
        self.encoder = TransformerEncoder(
            d_model=hyperparameters.hidden_size,
            d_ff=hyperparameters.d_ff,
            n_heads=hyperparameters.n_heads,
            n_layers=hyperparameters.num_layers,
            dropout=hyperparameters.dropout,
        )

        self.fc_offline = nn.Linear(hyperparameters.hidden_size, window_size * num_out_classes)

        #NOTE: Layers not used
        # self.positional_encoding = PositionalEncoding(
        #     d_model=hyperparameters.hidden_size,
        #     dropout=hyperparameters.dropout,
        #     max_len=window_size,
        # )

        # self.fc_online = nn.Linear(hyperparameters.hidden_size, num_out_classes)

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros(x.size(0), x.size(1)).to(x.device)

        x = self.input_projection(x)  # (B,L,D) -> (B,L,H)
        x = self.dynamic_positional_encoding(x) # (B,L,H) -> (B,L,H)

        x = self.encoder(x, mask)  # (B,L,H) -> (B,L,H)

        x = self.fc_online(x)  # (B,L,H) -> (B,L,D)
        return x
