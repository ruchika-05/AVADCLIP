
import torch
from pathlib import Path
from models.avadclip_model import AVadCLIPModel

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

filenames = [
    "Gladiator.2000__#00-13-32_00-14-40_label_A.mp4",
    "Election.2005__#00-05-30_00-06-56_label_A.mp4",
    "Mission.Impossible.II.2000__#01-49-35_01-51-52_label_B1-0-0.mp4"
]

model = AVadCLIPModel().to(device)

checkpoint = torch.load(
    "best_avadclip_model.pth",
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint)
model.eval()

print("\n===== FUSION COMPARISON =====")

for filename in filenames:

    stem = Path(filename).stem
    feature_path = (
        Path("features/multimodal")
        / f"{stem}_multimodal.pt"
    )

    data = torch.load(
        feature_path,
        map_location="cpu",
        weights_only=True
    )

    audio = data["audio"].float().unsqueeze(0).to(device)
    visual = data["visual"].float().unsqueeze(0).to(device)

    with torch.no_grad():

        visual_features = model.visual_temporal(visual)
        audio_features = model.audio_temporal(audio)

        fused, weights = model.fusion(
            visual_features, audio_features
        )

        score = model.anomaly_detector(fused).max().item()

        change = fused - visual_features

        visual_norm = torch.norm(
            visual_features, dim=-1
        ).mean().item()

        change_norm = torch.norm(
            change, dim=-1
        ).mean().item()

        gate = weights.mean().item()

    print(f"\nVideo: {filename.split('__')[0]}")
    print(f"Anomaly score: {score:.4f}")
    print(f"Visual norm: {visual_norm:.4f}")
    print(f"Fusion change norm: {change_norm:.4f}")
    print(f"Average fusion gate: {gate:.4f}")
