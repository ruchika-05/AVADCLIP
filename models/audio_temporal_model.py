import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_size,
        dilation,
        dropout=0.1,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = weight_norm(
            nn.Conv1d(
                input_channels,
                output_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                output_channels,
                output_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.network = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = (
            nn.Conv1d(input_channels, output_channels, kernel_size=1)
            if input_channels != output_channels
            else None
        )
        self.output_relu = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        return self.output_relu(self.network(x) + residual)


class AudioTemporalEncoder(nn.Module):
    """Three-level causal TCN used by the official AVadCLIP teacher."""

    def __init__(self, feature_dim=512, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.blocks = nn.Sequential(
            *[
                TemporalBlock(
                    input_channels=feature_dim,
                    output_channels=feature_dim,
                    kernel_size=2,
                    dilation=2**level,
                    dropout=dropout,
                )
                for level in range(3)
            ]
        )

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.ndim != 3:
            raise ValueError(
                "Expected audio features with shape [T, D] or [B, T, D], "
                f"got {tuple(x.shape)}"
            )
        if x.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Expected feature dimension {self.feature_dim}, "
                f"got {x.shape[-1]}"
            )

        return self.blocks(x.transpose(1, 2)).transpose(1, 2)
