import argparse
import csv
import zlib
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
import wav2clip

from models.audio_corruption import (
    CORRUPTION_CONDITIONS,
    SAMPLE_RATE,
    corrupt_waveform,
)


CONDITIONS = CORRUPTION_CONDITIONS


def _to_time_major(features):
    features = torch.as_tensor(features, dtype=torch.float32)
    if features.ndim == 3 and features.shape[0] == 1:
        features = features.squeeze(0).transpose(0, 1)
    if features.ndim != 2 or features.shape[1] != 512:
        raise ValueError(
            f"Expected Wav2CLIP features [T, 512], got {tuple(features.shape)}"
        )
    return features.contiguous()


def _align_sequence(features, target_length, name):
    difference = abs(features.shape[0] - target_length)
    if difference > 1:
        raise ValueError(
            f"{name} has {features.shape[0]} frames, expected "
            f"{target_length}; refusing a large temporal resample."
        )
    if difference:
        features = F.interpolate(
            features.transpose(0, 1).unsqueeze(0),
            size=target_length,
            mode="linear",
            align_corners=True,
        ).squeeze(0).transpose(0, 1)
    return features.contiguous()


def _seed_for_video(stem, condition):
    return zlib.crc32(f"{stem}:{condition}".encode("utf-8"))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate waveform-corrupted Wav2CLIP features for training "
            "manifest rows only."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("dataset_manifest_675_test134.csv"),
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=Path("features/audio"),
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=Path("features/multimodal"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("features/corruption_aware_train"),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue a partial output directory without replacing valid files.",
    )
    args = parser.parse_args()

    if not args.manifest.is_file():
        raise FileNotFoundError(f"Training manifest not found: {args.manifest}")

    train_rows = []
    with args.manifest.open(newline="", encoding="utf-8") as manifest_file:
        for row in csv.DictReader(manifest_file):
            if row.get("split") == "train":
                train_rows.append(row)
    if not train_rows:
        raise ValueError(
            f"No rows with split='train' were found in {args.manifest}"
        )

    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(
            f"Output directory already exists; use --resume only if it "
            f"contains this experiment's partial data: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    project_dir = Path(__file__).resolve().parents[1]
    torch.hub.set_dir(str(project_dir / ".cache" / "torch_hub"))
    encoder = wav2clip.get_model(
        frame_length=SAMPLE_RATE,
        hop_length=SAMPLE_RATE,
    )

    for index, row in enumerate(train_rows, start=1):
        stem = Path(row["filename"]).stem
        output_path = args.output_dir / f"{stem}_corruptions.pt"
        if output_path.is_file():
            if not args.resume:
                raise FileExistsError(f"Refusing to replace {output_path}")
            continue

        waveform_path = args.audio_dir / f"{stem}.wav"
        clean_feature_path = args.feature_dir / f"{stem}_multimodal.pt"
        if not waveform_path.is_file():
            raise FileNotFoundError(
                f"Training waveform is missing: {waveform_path}"
            )
        if not clean_feature_path.is_file():
            raise FileNotFoundError(
                f"Training clean features are missing: {clean_feature_path}"
            )

        clean_data = torch.load(
            clean_feature_path,
            map_location="cpu",
            weights_only=True,
        )
        visual_length = int(clean_data["visual"].shape[0])
        clean_audio_length = int(clean_data["audio"].shape[0])
        waveform, _ = librosa.load(
            waveform_path,
            sr=SAMPLE_RATE,
            mono=True,
        )

        variants = {}
        for condition in CONDITIONS:
            corrupted_waveform, frame_targets = corrupt_waveform(
                waveform,
                condition,
                _seed_for_video(stem, condition),
            )
            with torch.inference_mode():
                features = wav2clip.embed_audio(corrupted_waveform, encoder)
            features = _to_time_major(features)
            if abs(features.shape[0] - clean_audio_length) > 1:
                raise ValueError(
                    f"{stem}/{condition}: corrupted embedding length "
                    f"{features.shape[0]} differs from cached clean length "
                    f"{clean_audio_length} by more than one."
                )
            features = _align_sequence(
                features,
                visual_length,
                f"{stem}/{condition} audio features",
            )
            targets = torch.from_numpy(frame_targets).unsqueeze(0)
            targets = F.interpolate(
                targets.unsqueeze(0),
                size=visual_length,
                mode="linear",
                align_corners=True,
            ).squeeze(0).squeeze(0)
            variants[condition] = {
                "features": features.to(torch.float16),
                "targets": targets.to(torch.float32),
            }

        temporary_path = output_path.with_name(output_path.name + ".tmp")
        try:
            torch.save(variants, temporary_path)
            temporary_path.replace(output_path)
        finally:
            temporary_path.unlink(missing_ok=True)

        if index % 10 == 0 or index == len(train_rows):
            print(f"Prepared {index}/{len(train_rows)} training videos")

    print(
        f"Created waveform-corruption feature cache for {len(train_rows)} "
        f"training videos in {args.output_dir}"
    )


if __name__ == "__main__":
    main()
