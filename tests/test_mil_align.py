import torch

from models.mil_align import MILAlign, mil_topk_count

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

# One video, 16 temporal segments, 512-dimensional features
video_features = torch.randn(1, 16, 512, device=device)

# Seven category embeddings
class_features = torch.randn(1, 7, 512, device=device)

model = MILAlign(temperature=1.0).to(device)

output = model(video_features, class_features)

expected_k = mil_topk_count(video_features.shape[1])
expected_scores = torch.topk(
    output["alignment_map"],
    k=expected_k,
    dim=-1
).values.mean(dim=-1)

assert expected_k == 2
assert torch.allclose(output["class_scores"], expected_scores)
assert mil_topk_count(110) == 7
assert mil_topk_count(5) == 1
assert mil_topk_count(16, top_k=5) == 5

print("Alignment map:", output["alignment_map"].shape)
print("Class scores:", output["class_scores"].shape)
print("Probabilities:", output["probabilities"].shape)

print(
    "Probabilities sum:",
    output["probabilities"].sum(dim=1).item()
)

print("MIL-Align test successful!")
