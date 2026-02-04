"""
Python module implementing the Transformer model
"""

import os
import sys
import math
import torch
from torch import nn

sys.path.append(os.path.join(os.path.abspath(__file__),"ssm"))

from src.hyperparameters.transformer_hyperparameters import TransformerHyperparameters
from src.models.ssm.src.models import ModelHead

# Transformer Encoder
from transformer_encoder import TransformerEncoder
from transformer_encoder.utils import PositionalEncoding

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

        self.hparams = hyperparameters
        self.num_out_classes = num_out_classes

        self.input_projection = nn.Linear(num_features, hyperparameters.hidden_size)

        self.dynamic_positional_encoding = DynamicPositionalEncoding(
            d_model=self.hparams.hidden_size,
            dropout=self.hparams.dropout
        )
        self.encoder = TransformerEncoder(
            d_model=self.hparams.hidden_size,
            d_ff=self.hparams.d_ff,
            n_heads=self.hparams.n_heads,
            n_layers=self.hparams.num_layers,
            dropout=self.hparams.dropout,
        )

        self.head = ModelHead(
            hidden_size=self.hparams.hidden_size,
            n_layers = self.hparams.num_head_layers,
            k = self.hparams.k,
            dropout = self.hparams.dropout,
            output_size = self.num_out_classes,
            loss_type = self.hparams.loss_type,
            head_act = self.hparams.head_act,
            task = self.hparams.task,
            gap = self.hparams.gap
        )

        #NOTE: Layers not used
        # self.positional_encoding = PositionalEncoding(
        #     d_model=hyperparameters.hidden_size,
        #     dropout=hyperparameters.dropout,
        #     max_len=window_size,
        # )

        # self.fc_offline = nn.Linear(self.hparams.hidden_size, window_size * self.num_out_classes)
        # self.fc_online = nn.Linear(hyperparameters.hidden_size, num_out_classes)


    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros(x.size(0), x.size(1)).to(x.device)

        x = self.input_projection(x)  # (B,L,D) -> (B,L,H)
        x = self.dynamic_positional_encoding(x) # (B,L,H) -> (B,L,H)
        x = self.encoder(x, mask)  # (B,L,H) -> (B,L,H)
        x = self.head(x)  # (B,L,H) -> (B,L,D)

        return x
