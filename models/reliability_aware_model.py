import torch
import torch.nn as nn

from models.anomaly_detector import AnomalyDetector
from models.audio_temporal_model import AudioTemporalEncoder
from models.audio_visual_prompt import AudioVisualPrompt
from models.mil_align import MILAlign, mil_topk_count
from models.reliability_aware_fusion import ReliabilityAwareFusion
from models.reliability_estimator import AudioReliabilityEstimator
from models.temporal_model import VisualTemporalEncoder
from models.text_prompt import LearnableTextPrompt


class ReliabilityAwareAVadCLIPModel(nn.Module):
    def __init__(
        self,
        feature_dim=512,
        top_k=None,
        class_names=None,
        clip_model=None,
        n_ctx=4,
        reliability_hidden_dim=64,
        detach_reliability_for_fusion=False,
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

        self.visual_temporal = VisualTemporalEncoder(feature_dim)
        self.audio_temporal = AudioTemporalEncoder(feature_dim)
        self.reliability_estimator = AudioReliabilityEstimator(
            feature_dim,
            reliability_hidden_dim,
        )
        self.detach_reliability_for_fusion = detach_reliability_for_fusion
        self.fusion = ReliabilityAwareFusion(feature_dim)
        self.anomaly_detector = AnomalyDetector(feature_dim)
        self.audio_visual_prompt = AudioVisualPrompt(feature_dim)
        self.mil_align = MILAlign()
        self.text_prompt = (
            LearnableTextPrompt(clip_model, class_names, n_ctx)
            if clip_model is not None
            else None
        )
        self.top_k = top_k

    def train(self, mode=True):
        super().train(mode)
        if self.text_prompt is not None:
            self.text_prompt.clip_model.eval()
        return self

    def forward(self, visual, audio, class_features=None):
        visual_features = self.visual_temporal(visual)
        audio_reliability = self.reliability_estimator(audio)
        audio_features = self.audio_temporal(audio)
        fusion_reliability = (
            audio_reliability.detach()
            if self.detach_reliability_for_fusion
            else audio_reliability
        )
        fused_features, fusion_details = self.fusion(
            visual_features,
            audio_features,
            fusion_reliability,
        )

        segment_logits = self.anomaly_detector.forward_logits(fused_features)
        segment_scores = torch.sigmoid(segment_logits)
        k = mil_topk_count(segment_scores.shape[1], self.top_k)
        topk_scores = torch.topk(segment_scores, k=k, dim=1).values
        video_score = topk_scores.mean(dim=1)

        if class_features is None:
            if self.text_prompt is None:
                raise ValueError(
                    "Provide class_features or initialize the model with "
                    "clip_model."
                )
            class_features = self.text_prompt()

        prompt_output = self.audio_visual_prompt(
            fused_features,
            segment_logits,
            class_features,
        )
        alignment_output = self.mil_align(
            fused_features,
            prompt_output["enhanced_features"],
        )

        return {
            "segment_scores": segment_scores,
            "video_score": video_score,
            "fused_features": fused_features,
            "reliability_scores": audio_reliability,
            **fusion_details,
            "class_features": class_features,
            "enhanced_class_features": prompt_output["enhanced_features"],
            "global_features": prompt_output["global_features"],
            "class_weights": prompt_output["class_weights"],
            "alignment_map": alignment_output["alignment_map"],
            "class_scores": alignment_output["class_scores"],
            "class_probabilities": alignment_output["probabilities"],
        }
