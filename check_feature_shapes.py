
import csv
import torch
from pathlib import Path

MANIFEST = Path("dataset_manifest_675_test134.csv")

AUDIO_DIR = Path("features/audio_features")
VISUAL_DIR = Path("features/visual_features")

with open(MANIFEST, newline="") as f:
    videos = list(csv.DictReader(f))

print("Checking audio-visual feature shapes...\n")

mismatches = 0

for index, item in enumerate(videos, 1):

    name = Path(item["path"]).stem

    audio_path = AUDIO_DIR / f"{name}_audio.pt"
    visual_path = VISUAL_DIR / f"{name}_visual.pt"

    audio = torch.load(audio_path, map_location="cpu", weights_only=True)
    visual = torch.load(visual_path, map_location="cpu", weights_only=True)

    # Convert audio to [segments, 512]
    if audio.ndim == 3:
        audio = audio.squeeze(0).transpose(0, 1)

    elif audio.ndim != 2:
        print(f"[{index}] Unexpected audio shape: {audio.shape}")
        continue

    audio_segments = audio.shape[0]
    visual_segments = visual.shape[0]

    match = audio_segments == visual_segments

    if not match:
        mismatches += 1

    if index <= 10 or not match:
        print(f"[{index}] {name}")
        print("  Audio:", tuple(audio.shape))
        print("  Visual:", tuple(visual.shape))
        print("  Match:", match)

print("\n========== ALIGNMENT REPORT ==========")
print("Total videos:", len(videos))
print("Mismatched videos:", mismatches)
print("Matching videos:", len(videos) - mismatches)
