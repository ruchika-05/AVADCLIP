
import torch
from pathlib import Path
from models.avadclip_model import AVadCLIPModel

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

filename = "Gladiator.2000__#00-13-32_00-14-40_label_A.mp4"
stem = Path(filename).stem

data = torch.load(
    Path("features/multimodal") / f"{stem}_multimodal.pt",
    map_location="cpu",
    weights_only=True
)

audio = data["audio"].float().unsqueeze(0).to(device)
visual = data["visual"].float().unsqueeze(0).to(device)

model = AVadCLIPModel().to(device)

checkpoint = torch.load(
    "best_avadclip_model.pth",
    map_location=device,
    weights_only=True
)
model.load_state_dict(checkpoint)
model.eval()

with torch.no_grad():
    visual_features = model.visual_temporal(visual)
    audio_features = model.audio_temporal(audio)

    fused_features, weights = model.fusion(
        visual_features, audio_features
    )

    change = fused_features - visual_features

    visual_norm = torch.norm(visual_features, dim=-1).mean().item()
    audio_norm = torch.norm(audio_features, dim=-1).mean().item()
    change_norm = torch.norm(change, dim=-1).mean().item()

    print("Visual feature norm:", round(visual_norm, 4))
    print("Audio feature norm:", round(audio_norm, 4))
    print("Fusion change norm:", round(change_norm, 4))
    print(
        "Average fusion gate:",
        round(weights.mean().item(), 4),
    )
