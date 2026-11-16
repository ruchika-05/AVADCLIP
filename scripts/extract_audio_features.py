
import argparse
import csv
import torch
from pathlib import Path

from models.audio_encoder import AudioEncoder

# --------------------------------
# Paths
# --------------------------------

AUDIO_DIR = Path("features/audio")
OUTPUT_DIR = Path("features/audio_features")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--manifest",
    type=Path,
    default=Path("dataset_manifest_675_test134.csv"),
)
args = parser.parse_args()

# --------------------------------
# Load dataset
# --------------------------------

if not args.manifest.exists():
    raise FileNotFoundError(f"Manifest not found: {args.manifest}")

with open(args.manifest, newline="") as f:
    videos = list(csv.DictReader(f))

print("Total videos:", len(videos))

# Load encoder only once
print("\nLoading audio encoder...")
encoder = AudioEncoder()

successful = 0
skipped = 0
failed = []

# --------------------------------
# Extract features
# --------------------------------

for index, item in enumerate(videos, 1):

    video_path = Path(item["path"])
    video_name = video_path.stem

    audio_path = AUDIO_DIR / f"{video_name}.wav"
    output_path = OUTPUT_DIR / f"{video_name}_audio.pt"

    print(f"\n[{index}/{len(videos)}] {video_name}")

    try:
        if output_path.exists():
            if output_path.stat().st_size == 0:
                raise RuntimeError(f"Existing feature file is empty: {output_path}")
            print("  Feature file already exists; leaving it unchanged.")
            skipped += 1
            continue

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if audio_path.stat().st_size == 0:
            raise RuntimeError(f"Audio file is empty: {audio_path}")

        features = encoder.extract_features(str(audio_path))

        if not isinstance(features, torch.Tensor):
            features = torch.as_tensor(features)

        if features.numel() == 0:
            raise RuntimeError("Empty audio features!")

        temporary_path = output_path.with_name(f"{output_path.name}.tmp")
        try:
            torch.save(features.cpu(), temporary_path)
            temporary_path.replace(output_path)
        finally:
            temporary_path.unlink(missing_ok=True)

        print("  Feature shape:", tuple(features.shape))
        successful += 1

    except Exception as e:
        print("  ERROR:", e)
        failed.append((video_name, str(e)))

# --------------------------------
# Report
# --------------------------------

print("\n========== AUDIO FEATURE REPORT ==========")
print("Total videos:", len(videos))
print("Successful:", successful)
print("Skipped existing features:", skipped)
print("Failed:", len(failed))

if failed:
    print("\nFailed videos:")
    for name, error in failed:
        print(name, ":", error)

print("\nAudio feature extraction completed!")
