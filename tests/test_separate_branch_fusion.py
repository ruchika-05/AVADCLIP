import unittest

import torch

from models.avadclip_model import AVadCLIPModel
from models.fusion import AudioVisualFusion


class SeparateBranchFusionTests(unittest.TestCase):
    def test_joint_fusion_keeps_existing_parameter_names(self):
        fusion = AudioVisualFusion(feature_dim=8)

        self.assertEqual(
            set(fusion.state_dict()),
            {
                "weight_layer.weight",
                "weight_layer.bias",
                "residual_fc1.weight",
                "residual_fc1.bias",
                "residual_fc2.weight",
                "residual_fc2.bias",
            },
        )

    def test_separate_branches_preserve_feature_shape_and_details(self):
        fusion = AudioVisualFusion(feature_dim=8, architecture="separate")
        visual = torch.randn(2, 5, 8)
        audio = torch.randn(2, 5, 8)

        fused, details = fusion(visual, audio)

        self.assertEqual(fused.shape, visual.shape)
        self.assertEqual(details["visual_residual"].shape, visual.shape)
        self.assertEqual(details["audio_residual"].shape, audio.shape)
        self.assertTrue(
            torch.allclose(
                fused,
                visual + details["visual_residual"] + details["audio_residual"],
            )
        )
        self.assertNotIn("fusion_weights", details)

    def test_visual_residual_does_not_depend_on_audio(self):
        fusion = AudioVisualFusion(feature_dim=8, architecture="separate")
        visual = torch.randn(2, 5, 8)

        _, first_details = fusion(visual, torch.randn(2, 5, 8))
        _, second_details = fusion(visual, torch.randn(2, 5, 8))

        self.assertTrue(
            torch.equal(
                first_details["visual_residual"],
                second_details["visual_residual"],
            )
        )

    def test_separate_fusion_rejects_reliability_gate(self):
        fusion = AudioVisualFusion(feature_dim=8, architecture="separate")

        with self.assertRaisesRegex(ValueError, "not supported"):
            fusion(
                torch.randn(2, 5, 8),
                torch.randn(2, 5, 8),
                reliability=torch.ones(2),
            )

    def test_separate_model_rejects_audio_norm_reference(self):
        with self.assertRaisesRegex(ValueError, "does not support"):
            AVadCLIPModel(
                feature_dim=8,
                fusion_architecture="separate",
                audio_norm_reference=1.0,
            )

    def test_separate_model_forward_keeps_downstream_shapes(self):
        model = AVadCLIPModel(
            feature_dim=8,
            fusion_architecture="separate",
        )
        visual = torch.randn(2, 16, 8)
        audio = torch.randn_like(visual)
        class_features = torch.randn(7, 8)

        output = model(
            visual,
            audio,
            class_features=class_features,
        )

        self.assertEqual(output["fused_features"].shape, visual.shape)
        self.assertEqual(output["segment_scores"].shape, (2, 16))
        self.assertEqual(output["alignment_map"].shape, (2, 7, 16))
        self.assertEqual(output["visual_residual"].shape, visual.shape)
        self.assertEqual(output["audio_residual"].shape, audio.shape)
        self.assertNotIn("fusion_weights", output)

    def test_joint_model_keeps_legacy_fusion_weights_output(self):
        model = AVadCLIPModel(feature_dim=8)
        visual = torch.randn(2, 16, 8)
        output = model(
            visual,
            torch.randn_like(visual),
            class_features=torch.randn(7, 8),
        )

        self.assertEqual(output["fusion_weights"].shape, visual.shape)
        self.assertNotIn("visual_residual", output)


if __name__ == "__main__":
    unittest.main()
