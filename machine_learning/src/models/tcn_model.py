import numpy as np
import torch
from torch import nn

from src.models._tcn_layer import TCNLayer
from src.hyperparameters.cnn_hyperparameters import CNNHyperparameters 

class TCNModel(nn.Module):
    def __init__(self, hyperparameters: CNNHyperparameters, num_features: int, window_size: int, num_out_classes: int,
                 **_kwargs):
        super().__init__()
        self.num_features = num_features
        self.num_out_classes = num_out_classes
        self.hparams = hyperparameters

        self.tcn_layers = nn.Sequential(
            *[TCNLayer(kernel_size=self.hparams.kernel_size,
                      out_channels=int(self.hparams.initial_channel_size * self.hparams.channels_factor**layer_idx),
                      conv_stride=self.hparams.conv_stride,
                      max_pool_kernel_size=self.hparams.max_pool_size,
                      dropout=self.hparams.dropout,
                      activation=self.hparams.activation,
                      dilation=2**layer_idx
                ) for layer_idx in range(self.hparams.num_layers)],
        )

        self.fc = nn.LazyLinear(num_out_classes)

        #initializate the lazy modules
        dummy = torch.zeros((1, num_features, window_size))
        dummy = self.tcn_layers(dummy)
        if self.hparams.use_global_avg_pooling:
            dummy = torch.mean(dummy, dim=2)
        self.fc(dummy)
        

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.tcn_layers(x)

        #support both GAP and flattening
        if self.hparams.use_global_avg_pooling:
            x = torch.mean(x, dim=2)
        else:
            x = torch.flatten(x, start_dim=1)
        x = self.fc(x)

        return x