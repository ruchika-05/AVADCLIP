import torch
import torch.nn as nn
import torch.nn.functional as F


class ReliabilityAwareFusion(nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()
        self.visual_residual_fc1 = nn.Linear(feature_dim, feature_dim)
        self.visual_residual_fc2 = nn.Linear(feature_dim, feature_dim)
        self.audio_residual_fc1 = nn.Linear(feature_dim, feature_dim)
        self.audio_residual_fc2 = nn.Linear(feature_dim, feature_dim)

    def forward(self, visual, audio, reliability):
        if visual.shape != audio.shape:
            raise ValueError(
                "Visual and audio features must have matching shapes; "
                f"got {tuple(visual.shape)} and {tuple(audio.shape)}."
            )
        if reliability.shape != visual.shape[:-1]:
            raise ValueError(
                "Reliability must have one value per segment; "
                f"got {tuple(reliability.shape)} for features "
                f"{tuple(visual.shape)}."
            )

        visual_residual = self.visual_residual_fc2(
            F.gelu(self.visual_residual_fc1(visual))
        )
        audio_residual = self.audio_residual_fc2(
            F.gelu(self.audio_residual_fc1(audio))
        )
        gated_audio_residual = audio_residual * reliability.unsqueeze(-1)
        fused = visual + visual_residual + gated_audio_residual
        return fused, {
            "visual_residual": visual_residual,
            "audio_residual": audio_residual,
            "gated_audio_residual": gated_audio_residual,
        }
