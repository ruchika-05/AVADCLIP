
import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioVisualPrompt(nn.Module):

    def __init__(self, feature_dim=512):
        super().__init__()

        self.prompt_ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, av_features, anomaly_logits, class_features):
        """
        av_features:    [B, T, D]
        anomaly_logits: [B, T] - anomaly confidence scores
        class_features: [C, D] or [B, C, D]

        Returns:
            enhanced_features: [B, C, D]
        """

        batch_size, time_steps, feature_dim = av_features.shape

        if class_features.dim() == 2:
            class_features = class_features.unsqueeze(0).expand(
                batch_size, -1, -1
            )

        # Step 1: Create the anomaly-weighted global video representation.
        global_features = torch.bmm(
            anomaly_logits.unsqueeze(1),
            av_features
        ).squeeze(1)

        global_features = F.normalize(global_features, dim=-1)

       # Step 2: Calculate similarity between each class
        # and the global video representation.
        normalized_classes = F.normalize(class_features, dim=-1)

        similarity = torch.sum(
            normalized_classes * global_features.unsqueeze(1),
            dim=-1
        )

        class_weights = torch.softmax(similarity, dim=-1)

        # Step 3: Generate a global audio-visual prompt for each class.
        multimodal_prompts = (
            class_weights.unsqueeze(-1) *
            global_features.unsqueeze(1)
        )

        # Step 4: Add the prompt to the original class representation.
        prompt_input = class_features + multimodal_prompts

        # Step 5: Apply the FFN and residual connection.
        enhanced_features = (
            self.prompt_ffn(prompt_input) + class_features
        )

        return {
            "enhanced_features": enhanced_features,
            "global_features": global_features,
            "class_weights": class_weights,
            "multimodal_prompts": multimodal_prompts
        }
