
import torch
from pathlib import Path

from models.avadclip_model import AVadCLIPModel


def main():

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    filename = "Gladiator.2000__#00-13-32_00-14-40_label_A.mp4"

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

    model = AVadCLIPModel().to(device)

    model.load_state_dict(
        torch.load(
            "best_avadclip_model.pth",
            map_location=device,
            weights_only=True
        )
    )

    model.eval()

    with torch.no_grad():

        original = model(visual, audio)["video_score"].item()

        video_only = model(
            visual,
            torch.zeros_like(audio)
        )["video_score"].item()

        audio_only = model(
            torch.zeros_like(visual),
            audio
        )["video_score"].item()

    print("\n===== MODALITY ABLATION RESULTS =====")

    print(f"Original audio + video: {original:.4f}")
    print(f"Video only:             {video_only:.4f}")
    print(f"Audio only:             {audio_only:.4f}")


if __name__ == "__main__":
    main()
