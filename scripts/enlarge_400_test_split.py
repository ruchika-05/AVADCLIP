import argparse
from pathlib import Path

import pandas as pd


ADDITIONAL_TEST_MOVIES = {
    "Bullet.in.the.Head.1990",
    "Black.Hawk.Down.2001",
    "Fast.Five.2011",
    "Lord.of.War",
}


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create a new 400-video manifest with an approximately "
            "80-video source-movie-disjoint test split."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("dataset_manifest_400_split.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset_manifest_400_test80.csv"),
    )
    args = parser.parse_args()

    if not args.manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")
    if args.output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing manifest: {args.output}"
        )

    manifest = pd.read_csv(args.manifest)
    required_columns = {
        "filename",
        "split",
        "binary_label",
        "source_movie",
    }
    missing_columns = required_columns.difference(manifest.columns)
    if missing_columns:
        raise ValueError(
            f"Manifest is missing columns: {sorted(missing_columns)}"
        )
    if len(manifest) != 400 or manifest["filename"].duplicated().any():
        raise ValueError(
            "Expected 400 rows with unique filenames in the source manifest."
        )

    source_split_counts = manifest.groupby("source_movie")["split"].nunique()
    if (source_split_counts > 1).any():
        raise ValueError("Source manifest already has source-movie leakage.")

    selected_splits = manifest.loc[
        manifest["source_movie"].isin(ADDITIONAL_TEST_MOVIES),
        "split",
    ]
    if selected_splits.empty:
        raise ValueError("None of the selected source movies were found.")
    if set(selected_splits) != {"train"}:
        raise ValueError(
            "All selected movies must currently belong to train; found "
            f"{sorted(set(selected_splits))}."
        )

    updated = manifest.copy()
    selected = updated["source_movie"].isin(ADDITIONAL_TEST_MOVIES)
    updated.loc[selected, "split"] = "test"

    final_source_split_counts = updated.groupby("source_movie")["split"].nunique()
    if (final_source_split_counts > 1).any():
        raise RuntimeError("Source-movie leakage detected after reassignment.")

    test = updated[updated["split"] == "test"]
    if not 75 <= len(test) <= 85:
        raise RuntimeError(
            f"Expected approximately 80 test videos, got {len(test)}."
        )
    if set(test["binary_label"].astype(int).unique()) != {0, 1}:
        raise RuntimeError("Expanded test split must contain both binary classes.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = args.output.with_name(f"{args.output.name}.tmp")
    try:
        updated.to_csv(temporary_path, index=False)
        temporary_path.replace(args.output)
    finally:
        temporary_path.unlink(missing_ok=True)

    print(f"New split manifest: {args.output}")
    print(f"Videos: {len(updated)}")
    print("Split counts:", updated["split"].value_counts().to_dict())
    print(
        "Binary label counts in test:",
        test["binary_label"].astype(int).value_counts().sort_index().to_dict(),
    )
    print("Test source movies:", test["source_movie"].nunique())
    print(
        "Additional source movies moved from train:",
        sorted(ADDITIONAL_TEST_MOVIES),
    )
    print("Source movies crossing splits:", int((final_source_split_counts > 1).sum()))
    print(
        "\nImportant: models trained with the previous manifest have seen clips "
        "from the reassigned movies. Retrain using this new manifest before "
        "treating its test split as held out."
    )


if __name__ == "__main__":
    main()
