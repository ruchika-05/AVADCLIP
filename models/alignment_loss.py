import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mil_align import mil_topk_count


class MILAlignLoss(nn.Module):
    def __init__(self, num_classes=7, gamma=2.0):
        super().__init__()

        self.gamma = gamma

        # Learnable category weights, as in the official implementation
        self.alpha = nn.Parameter(
            torch.ones(num_classes) * 0.25
        )

    def forward(self, alignment_map, labels, lengths=None):
        """
        alignment_map: [B, C, T]
        labels:        [B, C] (multi-hot category labels)
        lengths:       [B] valid temporal lengths
        """

        batch_size, num_classes, max_length = alignment_map.shape

        if lengths is None:
            lengths = [max_length] * batch_size

        # Normalize multi-label targets
        labels = labels.float()
        labels = labels / labels.sum(dim=1, keepdim=True).clamp_min(1e-8)

        instance_logits = []

        for i in range(batch_size):
            length = int(lengths[i])
            length = min(length, max_length)

            k = mil_topk_count(length)

            valid_scores = alignment_map[i, :, :length]

            topk_values = torch.topk(
                valid_scores,
                k=k,
                dim=-1
            ).values

            instance_logits.append(topk_values.mean(dim=-1))

        instance_logits = torch.stack(instance_logits)

        # NCE: soft-label cross-entropy
        log_probs = F.log_softmax(instance_logits, dim=1)
        nce_loss = -torch.mean(
            torch.sum(labels * log_probs, dim=1)
        )

        # Focal loss
        probs = F.softmax(instance_logits, dim=1)
        pt = torch.sum(labels * probs, dim=1)

        alpha_per_sample = torch.sum(
            labels * self.alpha.unsqueeze(0),
            dim=1
        )

        focal_loss = -alpha_per_sample * (
            (1 - pt) ** self.gamma
        ) * torch.sum(labels * log_probs, dim=1)

        focal_loss = focal_loss.mean()

        alignment_loss = (nce_loss + focal_loss) / 2

        return {
            "alignment_loss": alignment_loss,
            "nce_loss": nce_loss,
            "focal_loss": focal_loss,
            "instance_logits": instance_logits
        }
