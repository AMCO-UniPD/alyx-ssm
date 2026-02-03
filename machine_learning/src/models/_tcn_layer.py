from torch import nn

class CNNLayer(nn.Module):
    def __init__(self,
                 kernel_size: int, out_channels: int,
                 conv_stride: int, max_pool_kernel_size: int, dropout: float,
                 activation: str, dilation: int):
        super().__init__()

        activation_klass = getattr(nn, activation)

        self.conv1 = nn.Conv1d(in_channels = None, #lazy init
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=conv_stride,
                               padding = (kernel_size - 1) * dilation,
                               dilation=dilation)
        self.act1 = activation_klass()
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=conv_stride,
                               padding = (kernel_size - 1) * dilation,
                               dilation=dilation)
        self.act2 = activation_klass()
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout(dropout)

        #residual connection
        self.downsample = nn.LazyConv1d(out_channels, kernel_size=1) 
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)
        out = self.bn1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.act2(out)
        out = self.bn2(out)
        out = self.dropout2(out)

        res = x if x.shape[1] == out.shape[1] else self.downsample(x)

        return out + res
