
import torch
import torch.nn as nn


class _QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class AnomalyDetector(nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()
        self.feature_mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            _QuickGELU(),
            nn.Linear(feature_dim * 4, feature_dim),
        )
        self.classifier = nn.Linear(feature_dim, 1)

    def forward_logits(self, x):
        return self.classifier(x + self.feature_mlp(x)).squeeze(-1)

    def forward(self, x):
        logits = self.forward_logits(x)
        return torch.sigmoid(logits)
