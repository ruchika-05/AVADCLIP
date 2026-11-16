import unittest

import torch

from models.fusion import AudioVisualFusion


class ReliabilityFusionTests(unittest.TestCase):
    def test_zero_reliability_disables_audio_residual(self):
        fusion = AudioVisualFusion(feature_dim=4)
        visual = torch.randn(2, 3, 4)
        audio = torch.randn(2, 3, 4)

        fused, weights = fusion(
            visual,
            audio,
            reliability=torch.zeros(2),
        )

        self.assertTrue(torch.equal(fused, visual))
        self.assertEqual(
            torch.count_nonzero(weights).item(),
            0,
        )

    def test_reliability_scales_fusion_weights(self):
        fusion = AudioVisualFusion(feature_dim=4)
        visual = torch.randn(2, 3, 4)
        audio = torch.randn(2, 3, 4)

        _, full_weights = fusion(visual, audio)
        _, half_weights = fusion(
            visual,
            audio,
            reliability=torch.full((2,), 0.5),
        )

        self.assertTrue(
            torch.allclose(
                half_weights,
                full_weights * 0.5,
            )
        )


if __name__ == "__main__":
    unittest.main()
