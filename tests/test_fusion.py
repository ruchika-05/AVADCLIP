
from pathlib import Path
import torch

from models.fusion import AudioVisualFusion

input_dir = Path("features/multimodal")
output_dir = Path("features/fused")
output_dir.mkdir(parents=True, exist_ok=True)

fusion_model = AudioVisualFusion(feature_dim=512)
fusion_model.eval()

total = 0
failed = 0

with torch.no_grad():

    for file_path in sorted(input_dir.glob("*_multimodal.pt")):

        try:
            data = torch.load(
                file_path,
                map_location="cpu",
                weights_only=True
            )

            visual = data["visual"]
            audio = data["audio"]

            fused_features, weights = fusion_model(
                visual, audio
            )

            # Validate output dimensions
            assert fused_features.shape == visual.shape
            assert weights.shape == visual.shape

            torch.save(
                {
                    "fused": fused_features,
                    "weights": weights,
                    "label": data["label"],
                    "filename": data["filename"]
                },
                output_dir / file_path.name
            )

            total += 1

            if total <= 5:
                print(f"\n{data['filename']}")
                print("Input shape:", visual.shape)
                print("Fused shape:", fused_features.shape)
                print("Weights shape:", weights.shape)
                print(
                    "Mean fusion weight:",
                    weights.mean().item()
                )

        except Exception as e:
            failed += 1
            print(f"FAILED: {file_path.name} | {e}")

print("\n========== FUSION REPORT ==========")
print("Total processed:", total)
print("Failed:", failed)

print("\nFusion testing completed!")
