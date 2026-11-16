
import torch
import torch.nn as nn
import torch.nn.functional as F


class VideoTextSimilarity(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.temperature = temperature

    def forward(self, video_features, text_features):

        # video_features: [B, T, 512]
        # text_features: [B, C, 512]

        video_features = F.normalize(
            video_features, dim=-1
        )

        text_features = F.normalize(
            text_features, dim=-1
        )

        # Compare every video segment with every text category
        similarities = torch.einsum(
            "btd,bcd->btc",
            video_features,
            text_features
        )

        return similarities / self.temperature
