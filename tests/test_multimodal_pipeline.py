
import torch

from models.temporal_model import VisualTemporalEncoder
from models.audio_temporal_model import AudioTemporalEncoder
from models.fusion import AudioVisualFusion

# Load one video's multimodal features
data = torch.load(
    "features/multimodal/sample/multimodal_features.pt",
    map_location="cpu",
    weights_only=True
)
visual = data["visual"]
audio = data["audio"]

print("Original visual shape:", visual.shape)
print("Original audio shape:", audio.shape)

# Initialize models
visual_encoder = VisualTemporalEncoder(feature_dim=512)
audio_encoder = AudioTemporalEncoder(feature_dim=512)
fusion_model = AudioVisualFusion(feature_dim=512)

visual_encoder.eval()
audio_encoder.eval()
fusion_model.eval()

# Process both modalities and fuse
with torch.no_grad():

    visual_temporal = visual_encoder(visual)
    audio_temporal = audio_encoder(audio)

    fused_features, fusion_weights = fusion_model(
        visual_temporal,
        audio_temporal
    )

print("\nVisual temporal shape:", visual_temporal.shape)
print("Audio temporal shape:", audio_temporal.shape)
print("Fused feature shape:", fused_features.shape)
print("Fusion weights shape:", fusion_weights.shape)

# Save results
# Save fused features
torch.save(
    {
        "fused": fused_features,
        "weights": fusion_weights
    },
    "features/multimodal/sample/temporally_fused_features.pt"
)

print("\nMultimodal pipeline completed!")
print("Features saved successfully!")
