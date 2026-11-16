
import csv
import cv2
import subprocess
from pathlib import Path
from collections import Counter

MANIFEST = "dataset_manifest.csv"

total = 0
readable = 0
failed = []
audio_counts = Counter()
label_counts = Counter()

with open(MANIFEST, newline="") as f:
    videos = list(csv.DictReader(f))

for index, item in enumerate(videos, 1):
    path = item["path"]

    label_counts[item["binary_label"]] += 1

    cap = cv2.VideoCapture(path)
    is_readable = cap.isOpened()

    if is_readable:
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if frames > 0 and fps > 0:
            readable += 1
        else:
            is_readable = False

    cap.release()

    if not is_readable:
        failed.append(path)

    # Check whether an audio stream exists
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=codec_type",
            "-of", "csv=p=0",
            path
        ],
        capture_output=True,
        text=True
    )

    has_audio = "audio" in result.stdout.lower()

    audio_counts["With audio" if has_audio else "Without audio"] += 1

    total += 1

    print(f"[{index}/{len(videos)}] {Path(path).name}: "
          f"{'OK' if is_readable else 'FAILED'}, "
          f"{'Audio' if has_audio else 'No audio'}")

print("\n========== DATASET REPORT ==========")
print("Total videos:", total)
print("Readable videos:", readable)
print("Failed videos:", len(failed))
print("Audio distribution:", audio_counts)
print("Binary label distribution:", label_counts)

if failed:
    print("\nFailed files:")
    for path in failed:
        print(path)
