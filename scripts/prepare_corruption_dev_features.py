import argparse
import csv
from pathlib import Path

import librosa
import torch
import torch.nn.functional as F
import wav2clip

from models.audio_corruption import SAMPLE_RATE, corrupt_waveform
from scripts.prepare_corruption_train_features import (
    CONDITIONS,
    _align_sequence,
    _seed_for_video,
    _to_time_major,
)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare waveform-corrupted features for development-set "
            "evaluation only. These features are not used for training."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("dataset_manifest_675_test134.csv"),
    )
    parser.add_argument("--audio-dir", type=Path, default=Path("features/audio"))
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=Path("features/multimodal"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("features/corruption_aware_dev_eval"),
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not args.manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")
    with args.manifest.open(newline="", encoding="utf-8") as manifest_file:
        val_rows = [
            row for row in csv.DictReader(manifest_file)
            if row.get("split") == "val"
        ]
    if not val_rows:
        raise ValueError(f"No development rows found in {args.manifest}")
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(
            f"Output directory exists; use --resume only for this evaluation: "
            f"{args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    project_dir = Path(__file__).resolve().parents[1]
    torch.hub.set_dir(str(project_dir / ".cache" / "torch_hub"))
    encoder = wav2clip.get_model(
        frame_length=SAMPLE_RATE,
        hop_length=SAMPLE_RATE,
    )

    for index, row in enumerate(val_rows, start=1):
        stem = Path(row["filename"]).stem
        destination = args.output_dir / f"{stem}_corruptions.pt"
        if destination.is_file():
            if not args.resume:
                raise FileExistsError(f"Refusing to replace {destination}")
            continue

        waveform_path = args.audio_dir / f"{stem}.wav"
        clean_feature_path = args.feature_dir / f"{stem}_multimodal.pt"
        if not waveform_path.is_file():
            raise FileNotFoundError(
                f"Development waveform is missing: {waveform_path}"
            )
        if not clean_feature_path.is_file():
            raise FileNotFoundError(
                f"Development clean features are missing: {clean_feature_path}"
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
            variant = {"features": features.to(torch.float16)}
            if condition == "partial_mute_50":
                mute_mask = torch.from_numpy(frame_targets).float().unsqueeze(0)
                mute_mask = F.interpolate(
                    mute_mask.unsqueeze(0),
                    size=visual_length,
                    mode="linear",
                    align_corners=True,
                ).squeeze(0).squeeze(0)
                variant["mute_mask"] = mute_mask <= 0.01
            variants[condition] = variant

        temporary = destination.with_name(destination.name + ".tmp")
        try:
            torch.save(variants, temporary)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)

        if index % 10 == 0 or index == len(val_rows):
            print(f"Prepared {index}/{len(val_rows)} development videos")

    print(
        f"Created waveform-corruption evaluation cache for {len(val_rows)} "
        f"development videos in {args.output_dir}"
    )


if __name__ == "__main__":
    main()
