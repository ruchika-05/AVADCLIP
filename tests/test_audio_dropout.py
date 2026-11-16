import unittest
import torch

from train import apply_audio_degradation, apply_audio_dropout


class AudioDropoutTests(unittest.TestCase):
    def test_zero_probability_preserves_audio_tensor(self):
        audio = torch.randn(2, 8, 512)

        self.assertIs(apply_audio_dropout(audio, 0.0), audio)

    def test_full_probability_mutes_every_video(self):
        audio = torch.randn(2, 8, 512)

        muted = apply_audio_dropout(audio, 1.0)

        self.assertEqual(torch.count_nonzero(muted).item(), 0)
        self.assertGreater(torch.count_nonzero(audio).item(), 0)

    def test_partial_probability_drops_whole_video_audio(self):
        audio = torch.ones(32, 8, 512)
        torch.manual_seed(7)

        augmented = apply_audio_dropout(audio, 0.5)

        per_video_sums = augmented.sum(dim=(1, 2))
        valid_sums = (per_video_sums == 0) | (per_video_sums == 8 * 512)
        self.assertTrue(torch.all(valid_sums).item())
        self.assertTrue(torch.any(per_video_sums == 0).item())
        self.assertTrue(torch.any(per_video_sums == 8 * 512).item())

    def test_invalid_probability_is_rejected(self):
        for probability in (-0.1, 1.1):
            with self.subTest(probability=probability):
                with self.assertRaisesRegex(ValueError, "between 0 and 1"):
                    apply_audio_dropout(torch.ones(1, 8, 512), probability)

    def test_audio_degradation_uses_expected_scale_levels(self):
        audio = torch.ones(128, 8, 512)
        torch.manual_seed(7)

        degraded = apply_audio_degradation(audio, 1.0)
        per_video_scales = degraded[:, 0, 0]

        self.assertTrue(
            torch.all(
                torch.isin(
                    per_video_scales,
                    torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]),
                )
            ).item()
        )
        self.assertEqual(torch.unique(per_video_scales).numel(), 5)

    def test_zero_degradation_probability_preserves_audio_tensor(self):
        audio = torch.randn(2, 8, 512)

        self.assertIs(apply_audio_degradation(audio, 0.0), audio)


if __name__ == "__main__":
    unittest.main()
