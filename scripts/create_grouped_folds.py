import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def main():
    parser = argparse.ArgumentParser(
        description="Assign videos to source-movie-grouped cross-validation folds."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("dataset_manifest_200_split.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset_manifest_200_group_folds.csv"),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")

    manifest = pd.read_csv(args.manifest)
    required_columns = {"filename", "source_movie", "binary_label"}
    missing_columns = required_columns.difference(manifest.columns)
    if missing_columns:
        raise ValueError(
            f"Manifest is missing columns: {sorted(missing_columns)}"
        )
    if manifest[list(required_columns)].isna().any().any():
        raise ValueError("Manifest has missing filenames, source movies, or labels.")
    if manifest["filename"].duplicated().any():
        raise ValueError("Manifest contains duplicate filenames.")

    labels = manifest["binary_label"].astype(int)
    if set(labels.unique()) != {0, 1}:
        raise ValueError("Grouped stratification requires both binary classes.")
    if manifest["source_movie"].nunique() < args.folds:
        raise ValueError(
            f"Need at least {args.folds} unique source movies, found "
            f"{manifest['source_movie'].nunique()}."
        )

    splitter = StratifiedGroupKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.seed,
    )
    manifest["fold"] = -1

    for fold, (_, validation_indices) in enumerate(
        splitter.split(manifest, labels, groups=manifest["source_movie"])
    ):
        manifest.loc[validation_indices, "fold"] = fold

    if (manifest["fold"] < 0).any():
        raise RuntimeError("Some videos were not assigned to a fold.")

    group_fold_counts = manifest.groupby("source_movie")["fold"].nunique()
    if (group_fold_counts != 1).any():
        raise RuntimeError("Source-movie leakage detected across folds.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, index=False)

    print(f"Grouped fold manifest created: {args.output}")
    print(
        f"Videos: {len(manifest)} | source movies: "
        f"{manifest['source_movie'].nunique()} | folds: {args.folds}"
    )
    print("\nFold composition:")
    for fold in range(args.folds):
        subset = manifest[manifest["fold"] == fold]
        class_counts = subset["binary_label"].value_counts().to_dict()
        movie_count = subset["source_movie"].nunique()
        category_counts = {}
        if "categories" in subset:
            for categories in subset["categories"].fillna(""):
                for category in str(categories).split(";"):
                    category = category.strip()
                    if category:
                        category_counts[category] = (
                            category_counts.get(category, 0) + 1
                        )
        print(
            f"  Fold {fold}: videos={len(subset)}, movies={movie_count}, "
            f"normal={class_counts.get(0, 0)}, "
            f"anomalous={class_counts.get(1, 0)}, "
            f"categories={category_counts}"
        )

    print("\nCross-fold source-movie leakage:", int((group_fold_counts > 1).sum()))


if __name__ == "__main__":
    main()
