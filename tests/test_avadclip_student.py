import torch
import torch.nn as nn

from models.avadclip_student import (
    AVadCLIPStudent,
    normal_anomaly_prompt_regularization,
)
from models.uncertainty_distillation import (
    uncertainty_weighted_distillation_loss,
)


class SyntheticTransformer(nn.Module):
    def forward(self, x, attn_mask=None):
        return x


class SyntheticClipModel(nn.Module):
    def __init__(self, feature_dim=512, vocab_size=49408, context_length=77):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, feature_dim)
        self.positional_embedding = nn.Parameter(
            torch.zeros(context_length, feature_dim)
        )
        self.transformer = SyntheticTransformer()
        self.ln_final = nn.Identity()
        self.text_projection = nn.Parameter(torch.eye(feature_dim))
        self.register_buffer("attn_mask", torch.zeros(context_length, context_length))
        for parameter in self.parameters():
            parameter.requires_grad = False


def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    batch_size, num_segments, feature_dim, num_classes = 2, 16, 512, 7
    clip_model = SyntheticClipModel(feature_dim).to(device)
    model = AVadCLIPStudent(clip_model=clip_model).to(device)
    visual = torch.randn(
        batch_size, num_segments, feature_dim, device=device
    )
    teacher_features = torch.randn_like(visual)

    output = model(visual)
    loss = uncertainty_weighted_distillation_loss(
        teacher_features,
        output["student_features"],
        output["log_variance"],
    ) + output["video_score"].mean() + output["class_scores"].mean()
    loss += normal_anomaly_prompt_regularization(
        output["conditioned_class_features"]
    )
    loss.backward()

    assert output["student_features"].shape == (
        batch_size, num_segments, feature_dim
    )
    assert output["log_variance"].shape == (
        batch_size, num_segments, feature_dim
    )
    assert output["video_score"].shape == (batch_size,)
    assert output["alignment_map"].shape == (
        batch_size, num_classes, num_segments
    )
    assert output["conditioned_class_features"].shape == (
        batch_size, num_classes, feature_dim
    )
    assert model.visual_temporal.transformer[0].attention.in_proj_weight.grad is not None
    assert model.anomaly_detector.feature_mlp[0].weight.grad is not None
    assert (
        model.feature_estimator.feature_enhancement.bottlenecks[0]
        .conv[0]
        .weight.grad
        is not None
    )
    assert model.text_prompt.context.grad is not None
    assert model.text_feature_mlp[0].weight.grad is not None
    assert model.visual_temporal.transformer.__len__() == 1
    assert (
        model.visual_temporal.transformer[0].attention_window == 64
    )
    assert all(not parameter.requires_grad for parameter in clip_model.parameters())

    print("Student feature shape:", output["student_features"].shape)
    print("Log-variance shape:", output["log_variance"].shape)
    print("Video score shape:", output["video_score"].shape)
    print("Alignment map shape:", output["alignment_map"].shape)
    print("\nVisual-only student test successful!")


if __name__ == "__main__":
    main()
