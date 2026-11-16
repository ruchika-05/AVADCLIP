import unittest

import numpy as np
import torch

from models.audio_corruption import CORRUPTION_CONDITIONS, corrupt_waveform
from models.reliability_aware_fusion import ReliabilityAwareFusion
from models.reliability_estimator import AudioReliabilityEstimator
from scripts.relabel_corruption_train_cache import _feature_similarity_targets


class CorruptionAwareTests(unittest.TestCase):
    def test_waveform_corruptions_keep_length_and_produce_targets(self):
        waveform = np.linspace(-0.5, 0.5, 5 * 16000, dtype=np.float32)
        for condition in CORRUPTION_CONDITIONS:
            with self.subTest(condition=condition):
                corrupted, targets = corrupt_waveform(
                    waveform,
                    condition,
                    seed=17,
                )
                self.assertEqual(corrupted.shape, waveform.shape)
                self.assertEqual(targets.shape, (5,))
                self.assertTrue(np.all((targets >= 0.0) & (targets <= 1.0)))
                if condition == "silence":
                    self.assertTrue(np.all(corrupted == 0.0))
                    self.assertTrue(np.all(targets == 0.0))

    def test_embedding_targets_measure_similarity_and_zero_unusable_segments(self):
        clean = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        variant = {
            "features": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "targets": torch.tensor([1.0, 1.0]),
        }
        targets = _feature_similarity_targets(
            "noise_snr_20",
            clean,
            variant,
        )
        self.assertTrue(torch.allclose(targets, torch.tensor([1.0, 0.0])))

        partial = {
            "features": torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
            "targets": torch.tensor([0.0, 1.0]),
        }
        partial_targets = _feature_similarity_targets(
            "partial_mute_50",
            clean,
            partial,
        )
        self.assertTrue(torch.allclose(partial_targets, torch.tensor([0.0, 1.0])))

        silence_targets = _feature_similarity_targets(
            "silence",
            clean,
            variant,
        )
        self.assertTrue(torch.equal(silence_targets, torch.zeros(2)))

    def test_estimator_outputs_one_bounded_score_per_segment(self):
        estimator = AudioReliabilityEstimator()
        features = torch.randn(2, 7, 512)
        scores = estimator(features)
        self.assertEqual(tuple(scores.shape), (2, 7))
        self.assertTrue(torch.all((scores >= 0.0) & (scores <= 1.0)))

    def test_fusion_gates_only_audio_residual(self):
        fusion = ReliabilityAwareFusion(feature_dim=4)
        visual = torch.randn(1, 3, 4)
        audio = torch.randn(1, 3, 4)
        reliability = torch.zeros(1, 3)
        fused_without_audio, details = fusion(visual, audio, reliability)
        fused_with_audio, _ = fusion(visual, audio, torch.ones_like(reliability))
        self.assertTrue(
            torch.allclose(
                fused_without_audio,
                visual + details["visual_residual"],
            )
        )
        self.assertTrue(
            torch.allclose(
                fused_with_audio - fused_without_audio,
                details["audio_residual"],
            )
        )


if __name__ == "__main__":
    unittest.main()
