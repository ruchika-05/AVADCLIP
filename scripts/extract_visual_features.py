
import argparse
import csv
import torch
from pathlib import Path

from models.visual_encoder import VisualEncoder

# --------------------------------
# Paths
# --------------------------------

FRAME_ROOT = Path("features/visual")
OUTPUT_DIR = Path("features/visual_features")

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

# Load encoder once
print("\nLoading visual encoder...")
encoder = VisualEncoder()

successful = 0
skipped = 0
failed = []

# --------------------------------
# Extract features
# --------------------------------

for index, item in enumerate(videos, 1):

    video_path = Path(item["path"])
    video_name = video_path.stem

    frame_dir = FRAME_ROOT / video_name
    output_path = OUTPUT_DIR / f"{video_name}_visual.pt"

    print(f"\n[{index}/{len(videos)}] {video_name}")

    try:
        if output_path.exists():
            if output_path.stat().st_size == 0:
                raise RuntimeError(f"Existing feature file is empty: {output_path}")
            print("  Feature file already exists; leaving it unchanged.")
            skipped += 1
            continue

        if not frame_dir.exists():
            raise FileNotFoundError(f"Frame directory not found: {frame_dir}")

        image_paths = sorted(frame_dir.glob("*.jpg"))

        if not image_paths:
            raise RuntimeError("No extracted frames found!")

        if any(path.stat().st_size == 0 for path in image_paths):
            raise RuntimeError(f"Empty frame file found in {frame_dir}")

        features = encoder.extract_features(image_paths)

        if not isinstance(features, torch.Tensor):
            features = torch.as_tensor(features)

        if features.numel() == 0:
            raise RuntimeError("Empty visual features!")

        temporary_path = output_path.with_name(f"{output_path.name}.tmp")
        try:
            torch.save(features.cpu(), temporary_path)
            temporary_path.replace(output_path)
        finally:
            temporary_path.unlink(missing_ok=True)

        print("  Frames:", len(image_paths))
        print("  Feature shape:", tuple(features.shape))

        successful += 1

    except Exception as e:
        print("  ERROR:", e)
        failed.append((video_name, str(e)))

# --------------------------------
# Report
# --------------------------------

print("\n========== VISUAL FEATURE REPORT ==========")
print("Total videos:", len(videos))
print("Successful:", successful)
print("Skipped existing features:", skipped)
print("Failed:", len(failed))

if failed:
    print("\nFailed videos:")
    for name, error in failed:
        print(name, ":", error)

print("\nVisual feature extraction completed!")
