import torch
import torch.nn as nn

from models.alignment_loss import MILAlignLoss
from models.anomaly_detector import AnomalyDetector
from models.mil_align import MILAlign, mil_topk_count
from models.temporal_model import VisualTemporalEncoder
from models.text_prompt import LearnableTextPrompt
from models.uncertainty_distillation import UncertaintyFeatureEstimator


def normal_anomaly_prompt_regularization(class_features):
    if class_features.ndim == 2:
        class_features = class_features.unsqueeze(0)
    if class_features.ndim != 3:
        raise ValueError(
            "Expected class features with shape [B, C, D] or [C, D], "
            f"got {tuple(class_features.shape)}"
        )
    if class_features.shape[1] < 2:
        raise ValueError(
            "At least two class features are required for prompt regularization"
        )

    normalized_features = torch.nn.functional.normalize(
        class_features, dim=-1
    )
    normal_feature = normalized_features[:, :1, :]
    anomaly_features = normalized_features[:, 1:, :]
    cosine_similarities = torch.matmul(
        normal_feature, anomaly_features.transpose(1, 2)
    )
    return cosine_similarities.abs().mean()


class _QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class AVadCLIPStudent(nn.Module):
    def __init__(
        self,
        clip_model,
        class_names=None,
        feature_dim=512,
        top_k=None,
    ):
        super().__init__()

        if class_names is None:
            class_names = [
                "normal activity",
                "fighting",
                "shooting",
                "explosion",
                "car accident",
                "abuse",
                "riot",
            ]

        self.visual_temporal = VisualTemporalEncoder(
            feature_dim,
            transformer_layers=1,
            attention_window=64,
        )
        self.feature_estimator = UncertaintyFeatureEstimator(feature_dim)
        self.anomaly_detector = AnomalyDetector(feature_dim)
        self.text_prompt = LearnableTextPrompt(clip_model, class_names)
        self.text_feature_mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            _QuickGELU(),
            nn.Linear(feature_dim * 4, feature_dim),
        )
        self.mil_align = MILAlign()
        self.top_k = top_k

    def train(self, mode=True):
        super().train(mode)
        self.text_prompt.clip_model.eval()
        return self

    def forward(self, visual):
        visual_features = self.visual_temporal(visual)
        student_features, log_variance = self.feature_estimator(
            visual_features
        )
        segment_logits = self.anomaly_detector.forward_logits(
            student_features
        )
        segment_scores = torch.sigmoid(segment_logits)

        k = mil_topk_count(segment_scores.shape[1], self.top_k)
        video_score = torch.topk(
            segment_scores,
            k=k,
            dim=1,
        ).values.mean(dim=1)

        class_features = self.text_prompt().unsqueeze(0).expand(
            student_features.shape[0], -1, -1
        )
        attention_weights = segment_logits.unsqueeze(1)
        visual_context = torch.bmm(attention_weights, student_features)
        visual_context = torch.nn.functional.normalize(
            visual_context, dim=-1
        )
        conditioned_class_features = (
            class_features
            + visual_context.expand(-1, class_features.shape[1], -1)
        )
        conditioned_class_features = (
            conditioned_class_features
            + self.text_feature_mlp(conditioned_class_features)
        )
        alignment = self.mil_align(
            student_features,
            conditioned_class_features,
        )

        return {
            "student_features": student_features,
            "log_variance": log_variance,
            "segment_scores": segment_scores,
            "video_score": video_score,
            "class_features": class_features,
            "conditioned_class_features": conditioned_class_features,
            "alignment_map": alignment["alignment_map"],
            "class_scores": alignment["class_scores"],
            "class_probabilities": alignment["probabilities"],
        }
