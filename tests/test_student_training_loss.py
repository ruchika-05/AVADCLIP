import unittest
import torch
import torch.nn.functional as F

from train_student import binary_classification_loss


class StudentTrainingLossTests(unittest.TestCase):
    def test_unit_positive_weight_matches_binary_cross_entropy(self):
        scores = torch.tensor([0.2, 0.8])
        labels = torch.tensor([0.0, 1.0])

        actual = binary_classification_loss(scores, labels)
        expected = F.binary_cross_entropy(scores, labels)

        torch.testing.assert_close(actual, expected)

    def test_positive_weight_only_scales_anomalous_examples(self):
        scores = torch.tensor([0.2, 0.8])
        labels = torch.tensor([0.0, 1.0])
        positive_weight = 2.0

        actual = binary_classification_loss(
            scores,
            labels,
            positive_class_weight=positive_weight,
        )
        normal_loss = F.binary_cross_entropy(
            scores[:1],
            labels[:1],
            reduction="sum",
        )
        anomalous_loss = F.binary_cross_entropy(
            scores[1:],
            labels[1:],
            reduction="sum",
        )
        expected = (normal_loss + positive_weight * anomalous_loss) / 2

        torch.testing.assert_close(actual, expected)

    def test_positive_weight_must_be_positive(self):
        scores = torch.tensor([0.5])
        labels = torch.tensor([1.0])

        with self.assertRaisesRegex(ValueError, "greater than zero"):
            binary_classification_loss(
                scores,
                labels,
                positive_class_weight=0,
            )


if __name__ == "__main__":
    unittest.main()
