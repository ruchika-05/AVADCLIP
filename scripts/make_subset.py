
import zipfile
import random
import csv
import re
from pathlib import Path
from collections import Counter, defaultdict

# ---------------- CONFIGURATION ----------------

ZIP_PATH = Path.home() / "Downloads" / "1-1004.zip"

OUTPUT_DIR = Path("dataset/videos/train")
MANIFEST_PATH = Path("dataset_manifest.csv")

TOTAL_SAMPLES = 100
RANDOM_SEED = 42

# -----------------------------------------------

random.seed(RANDOM_SEED)

CODE_NAMES = {
    "B1": "Fighting",
    "B2": "Shooting",
    "B4": "Riot",
    "B5": "Abuse",
    "B6": "Car accident",
    "G": "Explosion"
}

def parse_video(filename):
    match = re.search(r"label_([A-Z0-9-]+)\.mp4$", filename)

    if not match:
        return None

    codes = match.group(1).split("-")

    if codes == ["A"]:
        return {
            "binary_label": 0,
            "categories": "Normal",
            "primary": "Normal"
        }

    actual_codes = [c for c in codes if c != "0"]

    return {
        "binary_label": 1,
        "categories": ";".join(
            CODE_NAMES[c] for c in actual_codes
        ),
        "primary": CODE_NAMES[actual_codes[0]]
    }


def allocate_samples(groups, target):
    """Allocate samples proportionally across primary categories."""

    total = sum(len(v) for v in groups.values())

    quotas = {
        k: target * len(v) / total
        for k, v in groups.items()
    }

    allocation = {
        k: min(int(v), len(groups[k]))
        for k, v in quotas.items()
    }

    remaining = target - sum(allocation.values())

    order = sorted(
        groups,
        key=lambda k: quotas[k] - int(quotas[k]),
        reverse=True
    )

    while remaining:
        progressed = False

        for k in order:
            if allocation[k] < len(groups[k]):
                allocation[k] += 1
                remaining -= 1
                progressed = True

                if remaining == 0:
                    break

        if not progressed:
            break

    return allocation


# Read ZIP entries without extracting everything
with zipfile.ZipFile(ZIP_PATH, "r") as archive:

    normal = []
    anomalous = defaultdict(list)

    for info in archive.infolist():

        if not info.filename.lower().endswith(".mp4"):
            continue

        filename = Path(info.filename).name
        labels = parse_video(filename)

        if labels is None:
            continue

        entry = {
            "zip_path": info.filename,
            "filename": filename,
            **labels
        }

        if labels["binary_label"] == 0:
            normal.append(entry)
        else:
            anomalous[labels["primary"]].append(entry)

    print("Available normal videos:", len(normal))
    print("Available anomalous videos:",
          sum(len(v) for v in anomalous.values()))

    # Preserve the original binary class ratio
    total = len(normal) + sum(len(v) for v in anomalous.values())

    normal_target = round(
        TOTAL_SAMPLES * len(normal) / total
    )

    anomaly_target = TOTAL_SAMPLES - normal_target

    if normal_target > len(normal):
        raise ValueError("Not enough normal videos.")

    if anomaly_target > sum(len(v) for v in anomalous.values()):
        raise ValueError("Not enough anomalous videos.")

    selected_normal = random.sample(normal, normal_target)

    allocation = allocate_samples(anomalous, anomaly_target)

    selected_anomalous = []

    for category, count in allocation.items():
        selected_anomalous.extend(
            random.sample(anomalous[category], count)
        )

    selected = selected_normal + selected_anomalous
    random.shuffle(selected)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest_rows = []

    for index, item in enumerate(selected, 1):

        output_path = OUTPUT_DIR / item["filename"]

        # Extract only selected videos
        with archive.open(item["zip_path"]) as source:
            with open(output_path, "wb") as destination:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    destination.write(chunk)

        manifest_rows.append({
            "filename": item["filename"],
            "path": str(output_path),
            "split": "train",
            "binary_label": item["binary_label"],
            "categories": item["categories"],
            "primary_category": item["primary"]
        })

        print(f"[{index}/{TOTAL_SAMPLES}] Extracted: {item['filename']}")

    with open(MANIFEST_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=manifest_rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

print("\nSubset creation complete!")
print("Total selected:", len(selected))
print("Normal:", normal_target)
print("Anomalous:", anomaly_target)
print("Category distribution:")

print(Counter(
    item["primary"] for item in selected
))

print("Manifest saved at:", MANIFEST_PATH)
print("Videos saved at:", OUTPUT_DIR)
