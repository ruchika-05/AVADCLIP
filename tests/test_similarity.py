
import torch

from models.similarity import VideoTextSimilarity

device = "mps" if torch.backends.mps.is_available() else "cpu"

# Dummy features for testing
video_features = torch.randn(1, 16, 512).to(device)
text_features = torch.randn(1, 4, 512).to(device)

model = VideoTextSimilarity().to(device)

with torch.no_grad():
    similarities = model(video_features, text_features)

print("Video features:", video_features.shape)
print("Text features:", text_features.shape)
print("Similarity output:", similarities.shape)

expected = torch.einsum(
    "btd,bcd->btc",
    torch.nn.functional.normalize(video_features, dim=-1),
    torch.nn.functional.normalize(text_features, dim=-1),
) / 0.07
assert torch.allclose(similarities, expected)

print("\nSimilarity test successful!")
