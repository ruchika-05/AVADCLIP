import torch
import torch.nn as nn

from models.similarity import VideoTextSimilarity


def mil_topk_count(length, top_k=None):
    if length <= 0:
        raise ValueError(f"Sequence length must be positive, got {length}")
    if top_k is not None:
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        return min(top_k, length)
    return min(int(length / 16 + 1), length)


class MILAlign(nn.Module):
    def __init__(self, top_m=None, temperature=1.0):
        super().__init__()

        self.top_m = top_m
        self.temperature = temperature
        self.similarity = VideoTextSimilarity()

    def forward(self, video_features, class_features):
        """
        video_features: [B, T, D]
        class_features: [B, C, D]

        Returns:
            alignment_map: [B, C, T]
            class_scores:   [B, C]
            probabilities:  [B, C]
        """

        similarities = self.similarity(
            video_features,
            class_features
        )

        # Rearrange to [B, C, T]:
        # each category has a similarity score for every segment
        alignment_map = similarities.transpose(1, 2)

        # Select the top-M temporal similarities for each category
        m = mil_topk_count(alignment_map.shape[-1], self.top_m)

        top_m_values = torch.topk(
            alignment_map,
            k=m,
            dim=-1
        ).values

        # Average the selected similarities
        class_scores = top_m_values.mean(dim=-1)

        # Temperature-scaled class probabilities
        probabilities = torch.softmax(
            class_scores / self.temperature,
            dim=-1
        )

        return {
            "alignment_map": alignment_map,
            "class_scores": class_scores,
            "probabilities": probabilities
        }
