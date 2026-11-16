import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F


parser = argparse.ArgumentParser()
parser.add_argument(
    "--manifest",
    type=Path,
    default=Path("dataset_manifest_675_test134.csv"),
)
args = parser.parse_args()

if not args.manifest.exists():
    raise FileNotFoundError(f"Manifest not found: {args.manifest}")

manifest = pd.read_csv(args.manifest)
required_columns = {
    "filename",
    "binary_label",
    "categories",
    "primary_category",
}
missing_columns = required_columns.difference(manifest.columns)
if missing_columns:
    raise ValueError(f"Manifest is missing columns: {sorted(missing_columns)}")

audio_dir = Path("features/audio_features")
visual_dir = Path("features/visual_features")
output_dir = Path("features/multimodal")
output_dir.mkdir(parents=True, exist_ok=True)

successful = 0
skipped = 0
resampled = 0
failed = []

for index, row in manifest.iterrows():
    filename = Path(row["filename"]).stem
    audio_path = audio_dir / f"{filename}_audio.pt"
    visual_path = visual_dir / f"{filename}_visual.pt"
    output_path = output_dir / f"{filename}_multimodal.pt"

    try:
        if output_path.exists():
            if output_path.stat().st_size == 0:
                raise RuntimeError(f"Existing multimodal file is empty: {output_path}")
            print(f"[{index + 1}/{len(manifest)}] {filename} | already exists; skipping")
            skipped += 1
            continue

        audio = torch.load(audio_path, map_location="cpu", weights_only=True)
        visual = torch.load(visual_path, map_location="cpu", weights_only=True)

        if audio.ndim == 3 and audio.shape[0] == 1:
            audio = audio.squeeze(0).transpose(0, 1)

        if audio.ndim != 2 or visual.ndim != 2:
            raise ValueError(
                f"Expected 2D audio and visual tensors, got "
                f"{tuple(audio.shape)} and {tuple(visual.shape)}"
            )
        if audio.shape[1] != 512 or visual.shape[1] != 512:
            raise ValueError(
                f"Expected 512-dimensional features, got "
                f"{tuple(audio.shape)} and {tuple(visual.shape)}"
            )
        if audio.shape[0] != visual.shape[0]:
            length_difference = abs(audio.shape[0] - visual.shape[0])
            if length_difference > 1:
                raise ValueError(
                    f"Audio/visual sequence lengths differ by more than "
                    f"one step: {audio.shape[0]} vs {visual.shape[0]}"
                )
            audio = F.interpolate(
                audio.transpose(0, 1).unsqueeze(0),
                size=visual.shape[0],
                mode="linear",
                align_corners=True,
            ).squeeze(0).transpose(0, 1)
            resampled += 1
            print(
                f"[{index + 1}/{len(manifest)}] {filename} | "
                f"resampled audio features to {visual.shape[0]} steps"
            )

        multimodal = {
            "audio": audio.float(),
            "visual": visual.float(),
            "label": int(row["binary_label"]),
            "categories": str(row["categories"]).split(";"),
            "primary_category": str(row["primary_category"]),
            "filename": filename,
        }

        temporary_path = output_path.with_name(f"{output_path.name}.tmp")
        try:
            torch.save(multimodal, temporary_path)
            temporary_path.replace(output_path)
        finally:
            temporary_path.unlink(missing_ok=True)

        successful += 1
        print(
            f"[{index + 1}/{len(manifest)}] {filename} | "
            f"shape: {tuple(audio.shape)}"
        )

    except Exception as error:
        failed.append((filename, str(error)))
        print(f"[{index + 1}/{len(manifest)}] FAILED: {filename} | {error}")

print("\n========== MULTIMODAL FEATURE REPORT ==========")
print("Total videos:", len(manifest))
print("Successful:", successful)
print("Skipped existing features:", skipped)
print("Audio sequences resampled by one step:", resampled)
print("Failed:", len(failed))

if failed:
    print("\nFailed videos:")
    for filename, error in failed:
        print(f"{filename}: {error}")
    raise SystemExit(1)
