import argparse
import csv
import random
import re
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd


CODE_NAMES = {
    "B1": "Fighting",
    "B2": "Shooting",
    "B4": "Riot",
    "B5": "Abuse",
    "B6": "Car accident",
    "G": "Explosion",
}
EXPECTED_EXISTING_VIDEOS = 200
ADDITIONAL_TRAIN = 126
ADDITIONAL_VAL = 32
ADDITIONAL_TEST = 42
BATCH_SIZE = 100
RANDOM_SEED = 43
MANIFEST_COLUMNS = [
    "filename",
    "path",
    "split",
    "binary_label",
    "categories",
    "primary_category",
    "source_movie",
    "batch",
    "archive_member",
]


def parse_labels(filename):
    match = re.search(
        r"label_([A-Z0-9-]+)\.mp4$",
        filename,
        re.IGNORECASE,
    )
    if match is None:
        return None

    codes = match.group(1).upper().split("-")
    if codes == ["A"]:
        categories = ["Normal"]
        return {
            "binary_label": 0,
            "categories": categories,
            "primary_category": "Normal",
        }

    codes = [code for code in codes if code != "0"]
    if not codes or any(code not in CODE_NAMES for code in codes):
        return None

    categories = list(dict.fromkeys(CODE_NAMES[code] for code in codes))
    return {
        "binary_label": 1,
        "categories": categories,
        "primary_category": categories[0],
    }


def write_manifest(path, rows):
    temporary_path = path.with_name(f"{path.name}.tmp")
    try:
        pd.DataFrame(rows, columns=MANIFEST_COLUMNS).to_csv(
            temporary_path,
            index=False,
        )
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def build_plan(existing_rows, archive, video_dir):
    if len(existing_rows) != EXPECTED_EXISTING_VIDEOS:
        raise ValueError(
            f"Expected {EXPECTED_EXISTING_VIDEOS} existing videos, "
            f"found {len(existing_rows)}."
        )

    source_splits = {}
    existing_names = set()
    for row in existing_rows:
        filename = Path(row["filename"]).name
        source_movie = str(
            row.get("source_movie") or Path(filename).stem.split("__")[0]
        )
        split = str(row["split"])
        if split not in {"train", "val"}:
            raise ValueError(
                f"Expected existing split 'train' or 'val', found "
                f"{split!r} for {filename}."
            )
        if source_movie in source_splits and source_splits[source_movie] != split:
            raise ValueError(
                f"Existing source movie {source_movie} crosses splits."
            )
        source_splits[source_movie] = split
        existing_names.add(filename)

    pools = {"train": [], "val": [], "test": []}
    seen_names = set()
    for info in archive.infolist():
        if not info.filename.lower().endswith(".mp4"):
            continue
        filename = Path(info.filename).name
        if filename in existing_names or filename in seen_names:
            continue
        seen_names.add(filename)

        labels = parse_labels(filename)
        if labels is None:
            continue

        source_movie = Path(filename).stem.split("__")[0]
        split = source_splits.get(source_movie, "test")
        pools[split].append(
            {
                "filename": filename,
                "archive_member": info.filename,
                "source_movie": source_movie,
                **labels,
            }
        )

    required = {
        "train": ADDITIONAL_TRAIN,
        "val": ADDITIONAL_VAL,
        "test": ADDITIONAL_TEST,
    }
    for split, count in required.items():
        if len(pools[split]) < count:
            raise ValueError(
                f"Need {count} unused clips for {split}, "
                f"found {len(pools[split])}."
            )

    if len(pools["test"]) != ADDITIONAL_TEST:
        raise ValueError(
            "The number of clips from unseen source movies changed: "
            f"expected {ADDITIONAL_TEST}, found {len(pools['test'])}. "
            "Review the split plan instead of silently resampling."
        )

    rng = random.Random(RANDOM_SEED)
    selected = []
    for split, count in required.items():
        candidates = pools[split]
        chosen = candidates if split == "test" else rng.sample(candidates, count)
        rng.shuffle(chosen)

        first_batch_count = count // 2
        for index, item in enumerate(chosen):
            batch = 1 if index < first_batch_count else 2
            selected.append(
                {
                    "filename": item["filename"],
                    "path": str(video_dir / item["filename"]),
                    "split": split,
                    "binary_label": item["binary_label"],
                    "categories": ";".join(item["categories"]),
                    "primary_category": item["primary_category"],
                    "source_movie": item["source_movie"],
                    "batch": batch,
                    "archive_member": item["archive_member"],
                }
            )

    rng.shuffle(selected)
    for batch in (1, 2):
        batch_rows = [row for row in selected if row["batch"] == batch]
        if len(batch_rows) != BATCH_SIZE:
            raise RuntimeError(
                f"Batch {batch} has {len(batch_rows)} videos, "
                f"expected {BATCH_SIZE}."
            )

    old_rows = []
    for row in existing_rows:
        filename = Path(row["filename"]).name
        source_movie = str(
            row.get("source_movie") or Path(filename).stem.split("__")[0]
        )
        old_rows.append(
            {
                "filename": filename,
                "path": row["path"],
                "split": row["split"],
                "binary_label": int(row["binary_label"]),
                "categories": row["categories"],
                "primary_category": row["primary_category"],
                "source_movie": source_movie,
                "batch": 0,
                "archive_member": "",
            }
        )

    all_rows = old_rows + selected
    names = [row["filename"] for row in all_rows]
    if len(names) != len(set(names)):
        raise ValueError("Selected data contains duplicate filenames.")

    split_by_source = {}
    for row in all_rows:
        source_movie = row["source_movie"]
        split = row["split"]
        if (
            source_movie in split_by_source
            and split_by_source[source_movie] != split
        ):
            raise ValueError(
                f"Source-movie leakage detected for {source_movie}."
            )
        split_by_source[source_movie] = split

    return all_rows, selected


def load_or_create_plan(manifest_path, existing_manifest_path, archive_path, video_dir):
    existing = pd.read_csv(existing_manifest_path).to_dict("records")
    if manifest_path.exists():
        planned = pd.read_csv(manifest_path).fillna("").to_dict("records")
        planned_names = {str(row["filename"]) for row in planned}
        existing_names = {str(row["filename"]) for row in existing}
        if not existing_names.issubset(planned_names):
            raise ValueError(
                f"Existing plan {manifest_path} does not contain every "
                "video from the source manifest."
            )
        additions = [row for row in planned if int(row["batch"]) in {1, 2}]
        if len(planned) != 400 or len(additions) != 200:
            raise ValueError(
                f"Existing plan must contain 400 rows and 200 additions; "
                f"found {len(planned)} and {len(additions)}."
            )
        print(f"Reusing existing plan: {manifest_path}")
        return planned

    with zipfile.ZipFile(archive_path) as archive:
        all_rows, _ = build_plan(existing, archive, video_dir)
    write_manifest(manifest_path, all_rows)
    print(f"Created 400-video split manifest: {manifest_path}")
    return all_rows


def extract_batch(rows, batch, archive_path):
    selected = [row for row in rows if int(row["batch"]) == batch]
    if len(selected) != BATCH_SIZE:
        raise ValueError(
            f"Expected {BATCH_SIZE} planned videos in batch {batch}, "
            f"found {len(selected)}."
        )

    output_manifest = Path(f"dataset_manifest_400_batch{batch}.csv")
    write_manifest(output_manifest, selected)
    print(f"Batch manifest saved: {output_manifest}")

    video_dir = Path("dataset/videos/train")
    video_dir.mkdir(parents=True, exist_ok=True)
    successful = 0
    skipped = 0
    failures = []

    with zipfile.ZipFile(archive_path) as archive:
        infos = {info.filename: info for info in archive.infolist()}
        for index, row in enumerate(selected, 1):
            destination = video_dir / row["filename"]
            info = infos.get(row["archive_member"])
            if info is None:
                failures.append(
                    (row["filename"], "Archive member is missing.")
                )
                continue
            if destination.exists():
                if destination.stat().st_size != info.file_size:
                    failures.append(
                        (
                            row["filename"],
                            "Existing video size differs from archive; "
                            "refusing to overwrite.",
                        )
                    )
                    continue
                skipped += 1
                print(
                    f"[{index}/{BATCH_SIZE}] {row['filename']} "
                    "| already exists; skipping"
                )
                continue

            temporary_path = destination.with_name(
                f"{destination.name}.tmp"
            )
            try:
                with archive.open(info) as source, temporary_path.open("wb") as target:
                    while True:
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        target.write(chunk)
                if temporary_path.stat().st_size != info.file_size:
                    raise IOError(
                        f"Extracted size mismatch for {row['filename']}."
                    )
                temporary_path.replace(destination)
                successful += 1
                print(f"[{index}/{BATCH_SIZE}] Extracted: {row['filename']}")
            except Exception as error:
                temporary_path.unlink(missing_ok=True)
                failures.append((row["filename"], str(error)))
                print(f"[{index}/{BATCH_SIZE}] FAILED: {error}")

    print("\n========== DATASET EXPANSION REPORT ==========")
    print("Batch:", batch)
    print("Planned videos:", len(selected))
    print("Extracted:", successful)
    print("Skipped existing:", skipped)
    print("Failed:", len(failures))
    print(
        "Split counts:",
        dict(Counter(row["split"] for row in selected)),
    )
    print(
        "Primary categories:",
        dict(Counter(row["primary_category"] for row in selected)),
    )
    if failures:
        raise SystemExit(
            "Some videos failed extraction. Fix the reported errors and "
            "rerun this batch; existing verified files will be skipped."
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plan a source-movie-safe 400-video manifest and extract one "
            "100-video batch without overwriting existing files."
        )
    )
    parser.add_argument(
        "--existing-manifest",
        type=Path,
        default=Path("dataset_manifest_200_split.csv"),
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path.home() / "Downloads" / "1-1004.zip",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("dataset_manifest_400_split.csv"),
    )
    parser.add_argument("--batch", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()

    if not args.existing_manifest.is_file():
        raise FileNotFoundError(
            f"Existing manifest not found: {args.existing_manifest}"
        )
    if not args.archive.is_file():
        raise FileNotFoundError(f"Archive not found: {args.archive}")

    rows = load_or_create_plan(
        args.manifest,
        args.existing_manifest,
        args.archive,
        Path("dataset/videos/train"),
    )
    extract_batch(rows, args.batch, args.archive)


if __name__ == "__main__":
    main()
