import torch
import torch.nn as nn


class AudioReliabilityEstimator(nn.Module):
    def __init__(self, feature_dim=512, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, audio):
        if audio.ndim not in (2, 3):
            raise ValueError(
                "Audio features must have shape [T, D] or [B, T, D], "
                f"got {tuple(audio.shape)}"
            )
        estimator_dtype = self.network[0].weight.dtype
        audio = audio.to(dtype=estimator_dtype)
        return self.network(audio).squeeze(-1)
