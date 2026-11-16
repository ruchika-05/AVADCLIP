
import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioVisualFusion(nn.Module):

    ARCHITECTURES = ("joint", "separate")

    def __init__(self, feature_dim=512, architecture="joint"):
        super().__init__()
        if architecture not in self.ARCHITECTURES:
            raise ValueError(
                f"Unknown fusion architecture {architecture!r}; "
                f"expected one of {self.ARCHITECTURES}."
            )

        self.architecture = architecture

        if architecture == "joint":
            combined_dim = feature_dim * 2

            self.weight_layer = nn.Linear(
                combined_dim,
                feature_dim
            )

            self.residual_fc1 = nn.Linear(
                combined_dim,
                feature_dim
            )

            self.residual_fc2 = nn.Linear(
                feature_dim,
                feature_dim
            )
        else:
            self.visual_residual_fc1 = nn.Linear(feature_dim, feature_dim)
            self.visual_residual_fc2 = nn.Linear(feature_dim, feature_dim)
            self.audio_residual_fc1 = nn.Linear(feature_dim, feature_dim)
            self.audio_residual_fc2 = nn.Linear(feature_dim, feature_dim)

    def forward(self, visual, audio, reliability=None):
        if visual.shape != audio.shape:
            raise ValueError(
                "Visual and audio features must have matching shapes; "
                f"got {tuple(visual.shape)} and {tuple(audio.shape)}."
            )
        if visual.ndim not in (2, 3):
            raise ValueError(
                "Fusion inputs must have shape [time, feature_dim] or "
                "[batch, time, feature_dim]; "
                f"got {tuple(visual.shape)}."
            )

        if self.architecture == "separate":
            if reliability is not None:
                raise ValueError(
                    "Reliability gating is not supported by separate fusion."
                )

            visual_residual = self.visual_residual_fc2(
                F.gelu(self.visual_residual_fc1(visual))
            )
            audio_residual = self.audio_residual_fc2(
                F.gelu(self.audio_residual_fc1(audio))
            )
            fused = visual + visual_residual + audio_residual
            return fused, {
                "visual_residual": visual_residual,
                "audio_residual": audio_residual,
            }

        combined = torch.cat((visual, audio), dim=-1)

        weights = torch.sigmoid(self.weight_layer(combined))
        if reliability is not None:
            if reliability.ndim == 1:
                reliability = reliability[:, None, None]
            elif reliability.ndim == 2:
                reliability = reliability[:, :, None]
            weights = weights * reliability

        residual = self.residual_fc2(
            F.gelu(self.residual_fc1(combined))
        )
        fused = visual + weights * residual

        return fused, weights
