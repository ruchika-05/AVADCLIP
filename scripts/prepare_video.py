
import argparse
import csv
import cv2
import subprocess
from pathlib import Path

# --------------------------------
# Project paths
# --------------------------------

FRAME_ROOT = Path("features/visual")
AUDIO_ROOT = Path("features/audio")

FRAME_ROOT.mkdir(parents=True, exist_ok=True)
AUDIO_ROOT.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--manifest",
    type=Path,
    default=Path("dataset_manifest_675_test134.csv"),
)
args = parser.parse_args()

# --------------------------------
# Read dataset manifest
# --------------------------------

if not args.manifest.exists():
    raise FileNotFoundError(f"Manifest not found: {args.manifest}")

with open(args.manifest, newline="") as f:
    videos = list(csv.DictReader(f))

print("Total videos to preprocess:", len(videos))

successful = 0
skipped_audio = 0
failed = []

# --------------------------------
# Process videos one at a time
# --------------------------------

for index, item in enumerate(videos, 1):

    video_path = Path(item["path"])
    video_name = video_path.stem

    frame_dir = FRAME_ROOT / video_name
    audio_path = AUDIO_ROOT / f"{video_name}.wav"

    print(f"\n[{index}/{len(videos)}] Processing: {video_name}")

    try:
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        frame_dir.mkdir(parents=True, exist_ok=True)

        # ----------------------------
        # 1. Extract video frames
        # ----------------------------

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise RuntimeError("Could not open video!")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0 or total_frames <= 0:
            cap.release()
            raise RuntimeError("Invalid video FPS or frame count.")

        duration = total_frames / fps
        num_segments = int(duration)

        if num_segments == 0:
            cap.release()
            raise RuntimeError("Video is shorter than one second.")

        extracted_frames = 0

        for i in range(num_segments):

            frame_path = frame_dir / f"frame_{i:03d}.jpg"
            if frame_path.is_file() and frame_path.stat().st_size > 0:
                extracted_frames += 1
                continue

            timestamp = i + 0.5
            frame_index = round(timestamp * fps)

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = cap.read()

            if success and cv2.imwrite(str(frame_path), frame):
                extracted_frames += 1

        cap.release()

        if extracted_frames == 0:
            raise RuntimeError("No frames extracted!")

        # ----------------------------
        # 2. Extract audio
        # ----------------------------

        if audio_path.exists():
            if audio_path.stat().st_size == 0:
                raise RuntimeError(f"Existing audio file is empty: {audio_path}")
            print("  Audio already exists; leaving it unchanged.")
            skipped_audio += 1
        else:
            command = [
                "ffmpeg",
                "-n",
                "-v", "error",
                "-i", str(video_path),
                "-vn",
                "-ac", "1",
                "-ar", "16000",
                "-c:a", "pcm_s16le",
                str(audio_path)
            ]

            result = subprocess.run(
                command,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                raise RuntimeError(result.stderr[-1000:])

            if not audio_path.exists() or audio_path.stat().st_size == 0:
                raise RuntimeError("Audio extraction produced an empty file.")

        print("  Duration:", round(duration, 2), "seconds")
        print("  Frames:", extracted_frames)
        print("  Audio:", audio_path)
        successful += 1

    except Exception as e:
        print("  ERROR:", e)
        failed.append((video_name, str(e)))

print("\n========== PREPROCESSING REPORT ==========")
print("Total videos:", len(videos))
print("Successful:", successful)
print("Audio files preserved:", skipped_audio)
print("Failed:", len(failed))

if failed:
    print("\nFailed videos:")
    for name, error in failed:
        print(name, ":", error)

print("\nPreprocessing completed!")
