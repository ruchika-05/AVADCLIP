
import torch

from models.anomaly_detector import AnomalyDetector

# Load temporal features
features = torch.load(
    "features/multimodal/sample/temporal_features.pt",
    map_location="cpu",
    weights_only=True
)

print("Input shape:", features.shape)

# Initialize anomaly detector
model = AnomalyDetector(feature_dim=512)

model.eval()

# Generate anomaly scores
with torch.no_grad():
    scores = model(features)

print("\nAnomaly detection completed!")

print("Output shape:", scores.shape)

print("\nAnomaly scores:")

for i, score in enumerate(scores[0]):
    print(
        f"Segment {i + 1}: {score.item():.4f}"
    )

print("\nMinimum score:", scores.min().item())
print("Maximum score:", scores.max().item())
