import torch

from models.uncertainty_distillation import (
    UncertaintyFeatureEstimator,
    uncertainty_weighted_distillation_loss,
)


def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    batch_size, num_segments, feature_dim = 2, 16, 512
    visual_features = torch.randn(
        batch_size, num_segments, feature_dim, device=device
    )
    teacher_features = torch.randn_like(visual_features)

    estimator = UncertaintyFeatureEstimator(feature_dim=feature_dim).to(device)
    student_features, log_variance = estimator(visual_features)
    loss = uncertainty_weighted_distillation_loss(
        teacher_features,
        student_features,
        log_variance,
    )
    loss.backward()

    assert student_features.shape == visual_features.shape
    assert log_variance.shape == visual_features.shape
    assert torch.isfinite(loss)
    assert (
        estimator.feature_enhancement.bottlenecks[0].conv[0].weight.grad
        is not None
    )
    assert estimator.uncertainty_head[0].weight.grad is not None

    print("Student feature shape:", student_features.shape)
    print("Log-variance shape:", log_variance.shape)
    print("Distillation loss:", loss.item())
    print("\nUncertainty distillation test successful!")


if __name__ == "__main__":
    main()
