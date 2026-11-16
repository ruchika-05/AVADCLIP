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
        av_features:      [B, T, D]
        anomaly_logits:   [B, T]
        class_features:   [C, D] or [B, C, D]

        Returns:
            enhanced_features: [B, C, D]
        """

        batch_size, _, _ = av_features.shape

        if class_features.dim() == 2:
            class_features = class_features.unsqueeze(0).expand(
                batch_size, -1, -1
            )

        global_features = torch.bmm(
            anomaly_logits.unsqueeze(1),
            av_features
        ).squeeze(1)
        global_features = F.normalize(global_features, dim=-1)

        global_by_class = global_features.unsqueeze(1).expand(
            -1, class_features.shape[1], -1
        )
        similarity = torch.bmm(
            class_features,
            global_by_class.transpose(1, 2)
        )
        class_attention = torch.softmax(similarity, dim=-1)
        multimodal_prompts = torch.bmm(class_attention, global_by_class)
        prompt_input = class_features + multimodal_prompts
        enhanced_features = prompt_input + self.prompt_ffn(prompt_input)

        return {
            "enhanced_features": enhanced_features,
            "global_features": global_features,
            "class_weights": class_attention,
            "multimodal_prompts": multimodal_prompts
        }