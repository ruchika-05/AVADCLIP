import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F

from models.audio_corruption import CORRUPTION_CONDITIONS


def _feature_similarity_targets(condition, clean_audio, variant):
    features = variant["features"].float()
    if features.shape != clean_audio.shape:
        raise ValueError(
            f"{condition} feature shape {tuple(features.shape)} does not "
            f"match clean shape {tuple(clean_audio.shape)}"
        )
    targets = F.cosine_similarity(features, clean_audio, dim=-1).clamp(
        min=0.0,
        max=1.0,
    )
    if condition == "silence":
        targets.zero_()
    elif condition == "partial_mute_50":
        targets = targets * variant["targets"].float()
    return targets


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create a training-only corruption cache with revised, "
            "embedding-preservation reliability targets without "
            "re-extracting embeddings."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("dataset_manifest_675_test134.csv"),
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("features/corruption_aware_train"),
    )
    parser.add_argument(
        "--clean-feature-dir",
        type=Path,
        default=Path("features/multimodal"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("features/corruption_aware_train_similarity"),
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not args.manifest.is_file():
        raise FileNotFoundError(f"Training manifest not found: {args.manifest}")
    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input corruption cache not found: {args.input_dir}")
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(
            f"Output directory exists; use --resume only for this experiment: "
            f"{args.output_dir}"
        )

    with args.manifest.open(newline="", encoding="utf-8") as manifest_file:
        train_stems = [
            Path(row["filename"]).stem
            for row in csv.DictReader(manifest_file)
            if row.get("split") == "train"
        ]
    if not train_stems:
        raise ValueError(f"No training rows found in {args.manifest}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, stem in enumerate(train_stems, start=1):
        source = args.input_dir / f"{stem}_corruptions.pt"
        destination = args.output_dir / source.name
        if destination.is_file():
            if not args.resume:
                raise FileExistsError(f"Refusing to replace {destination}")
            continue
        if not source.is_file():
            raise FileNotFoundError(f"Training corruption cache missing: {source}")

        clean_path = (
            args.clean_feature_dir / f"{stem}_multimodal.pt"
        )
        if not clean_path.is_file():
            raise FileNotFoundError(
                f"Training clean audio features missing: {clean_path}"
            )
        clean_data = torch.load(
            clean_path,
            map_location="cpu",
            weights_only=True,
        )
        clean_audio = clean_data["audio"].float()
        variants = torch.load(source, map_location="cpu", weights_only=True)
        if set(variants) != set(CORRUPTION_CONDITIONS):
            raise ValueError(
                f"{source} has unexpected conditions: {sorted(variants)}"
            )
        for condition in CORRUPTION_CONDITIONS:
            if condition == "partial_mute_50":
                variants[condition]["mute_mask"] = (
                    variants[condition]["targets"].float() <= 0.01
                )
            variants[condition]["targets"] = _feature_similarity_targets(
                condition,
                clean_audio,
                variants[condition],
            )

        temporary = destination.with_name(destination.name + ".tmp")
        try:
            torch.save(variants, temporary)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)

        if index % 20 == 0 or index == len(train_stems):
            print(f"Relabeled {index}/{len(train_stems)} training videos")

    print(f"Created embedding-preservation training cache in {args.output_dir}")


if __name__ == "__main__":
    main()
