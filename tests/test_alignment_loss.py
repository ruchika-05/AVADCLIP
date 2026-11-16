import torch

from models.alignment_loss import MILAlignLoss

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

# Synthetic alignment scores: 1 video, 7 categories, 16 segments
alignment_map = torch.randn(
    1, 7, 16, device=device, requires_grad=True
)

# Example with two anomaly categories
labels = torch.tensor(
    [[0, 0, 1, 0, 1, 0, 0]],
    dtype=torch.float32,
    device=device
)

criterion = MILAlignLoss(num_classes=7).to(device)

output = criterion(alignment_map, labels, lengths=[16])

output["alignment_loss"].backward()

print("NCE loss:", output["nce_loss"].item())
print("Focal loss:", output["focal_loss"].item())
print("Combined loss:", output["alignment_loss"].item())
print("Alignment gradients available:", alignment_map.grad is not None)
print("Alpha gradients available:", criterion.alpha.grad is not None)
