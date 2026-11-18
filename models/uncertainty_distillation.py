
import torch
import torch.nn as nn


class _FeatureEnhancementBranch(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(
                feature_dim,
                feature_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        return x + self.conv(x)


class UncertaintyFeatureEstimator(nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()

        self.feature_enhancement = _FeatureEnhancementBranch(feature_dim)

        # Three-layer CNN for log-variance prediction.
        self.uncertainty_head = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=1),
        )

    def forward(self, visual_features):
        if visual_features.ndim != 3:
            raise ValueError(
                "Expected visual features with shape [B, T, D], "
                f"got {tuple(visual_features.shape)}"
            )

        channels_first = visual_features.transpose(1, 2)

        mean = self.feature_enhancement(
            channels_first
        ).transpose(1, 2)

        log_variance = self.uncertainty_head(
            channels_first
        ).transpose(1, 2)

        return mean, log_variance


def uncertainty_weighted_distillation_loss(
    teacher_features,
    student_features,
    log_variance,
):
    if teacher_features.shape != student_features.shape:
        raise ValueError(
            "Teacher and student feature shapes must match, got "
            f"{tuple(teacher_features.shape)} and "
            f"{tuple(student_features.shape)}"
        )

    if teacher_features.shape != log_variance.shape:
        raise ValueError(
            "Log-variance shape must match teacher features, got "
            f"{tuple(log_variance.shape)} and "
            f"{tuple(teacher_features.shape)}"
        )

    squared_error = (teacher_features - student_features).square()
    variance = log_variance.exp()

    loss = squared_error / (variance + 1e-6) + log_variance

    return loss.mean()
