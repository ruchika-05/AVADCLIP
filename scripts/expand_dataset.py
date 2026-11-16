
import zipfile
import random
import csv
import re
from pathlib import Path
from collections import defaultdict, Counter

ZIP_PATH = Path.home() / "Downloads" / "1-1004.zip"
OLD_MANIFEST = Path("dataset_manifest.csv")
NEW_MANIFEST = Path("dataset_manifest_200.csv")
OUTPUT_DIR = Path("dataset/videos/train")

ADDITIONAL_SAMPLES = 100
RANDOM_SEED = 43

CODE_NAMES = {
    "B1": "Fighting",
    "B2": "Shooting",
    "B4": "Riot",
    "B5": "Abuse",
    "B6": "Car accident",
    "G": "Explosion"
}

random.seed(RANDOM_SEED)


def parse_video(filename):
    match = re.search(
        r"label_([A-Z0-9-]+)\.mp4$",
        filename,
        re.IGNORECASE
    )

    if not match:
        return None

    codes = match.group(1).upper().split("-")

    if codes == ["A"]:
        return {
            "binary_label": 0,
            "categories": "Normal",
            "primary": "Normal"
        }

    actual_codes = [code for code in codes if code != "0"]

    if not actual_codes or any(code not in CODE_NAMES for code in actual_codes):
        return None

    categories = list(dict.fromkeys(
        CODE_NAMES[code] for code in actual_codes
    ))

    return {
        "binary_label": 1,
        "categories": ";".join(categories),
        "primary": categories[0]
    }


def allocate_samples(groups, target):
    total = sum(len(v) for v in groups.values())

    if target > total:
        raise ValueError("Not enough anomalous videos.")

    quotas = {
        key: target * len(values) / total
        for key, values in groups.items()
    }

    allocation = {
        key: int(quotas[key])
        for key in groups
    }

    remaining = target - sum(allocation.values())

    order = sorted(
        groups,
        key=lambda key: quotas[key] - int(quotas[key]),
        reverse=True
    )

    while remaining:
        for key in order:
            if allocation[key] < len(groups[key]):
                allocation[key] += 1
                remaining -= 1

                if remaining == 0:
                    break

    return allocation


if not ZIP_PATH.exists():
    raise FileNotFoundError(f"ZIP not found: {ZIP_PATH}")

if not OLD_MANIFEST.exists():
    raise FileNotFoundError(f"Existing manifest not found: {OLD_MANIFEST}")

with open(OLD_MANIFEST, newline="") as file:
    existing_rows = list(csv.DictReader(file))

existing_names = {row["filename"] for row in existing_rows}

print("Existing videos:", len(existing_names))

normal = []
anomalous = defaultdict(list)
seen_names = set()

with zipfile.ZipFile(ZIP_PATH, "r") as archive:

    for info in archive.infolist():
        if not info.filename.lower().endswith(".mp4"):
            continue

        filename = Path(info.filename).name

        if filename in existing_names or filename in seen_names:
            continue

        seen_names.add(filename)

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

    available_anomalous = sum(len(v) for v in anomalous.values())
    available_total = len(normal) + available_anomalous

    print("Available additional normal:", len(normal))
    print("Available additional anomalous:", available_anomalous)

    if available_total < ADDITIONAL_SAMPLES:
        raise ValueError("Not enough unused videos in the archive.")

    normal_target = round(
        ADDITIONAL_SAMPLES * len(normal) / available_total
    )
    anomaly_target = ADDITIONAL_SAMPLES - normal_target

    if normal_target > len(normal):
        raise ValueError("Not enough unused normal videos.")

    selected = random.sample(normal, normal_target)

    allocation = allocate_samples(anomalous, anomaly_target)

    for category, count in allocation.items():
        selected.extend(random.sample(anomalous[category], count))

    random.shuffle(selected)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    new_rows = []

    for index, item in enumerate(selected, 1):
        output_path = OUTPUT_DIR / item["filename"]

        if output_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing video: {output_path}"
            )

        with archive.open(item["zip_path"]) as source:
            with open(output_path, "wb") as destination:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    destination.write(chunk)

        new_rows.append({
            "filename": item["filename"],
            "path": str(output_path),
            "split": "train",
            "binary_label": item["binary_label"],
            "categories": item["categories"],
            "primary_category": item["primary"]
        })

        print(
            f"[{index}/{ADDITIONAL_SAMPLES}] "
            f"Extracted: {item['filename']}"
        )

all_rows = existing_rows + new_rows

with open(NEW_MANIFEST, "w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "filename",
            "path",
            "split",
            "binary_label",
            "categories",
            "primary_category"
        ]
    )
    writer.writeheader()
    writer.writerows(all_rows)

print("\nDataset expansion complete!")
print("Previously existing videos:", len(existing_rows))
print("New videos:", len(new_rows))
print("Total videos:", len(all_rows))
print("New video category distribution:")
print(Counter(row["primary_category"] for row in new_rows))
print("Expanded manifest:", NEW_MANIFEST)
