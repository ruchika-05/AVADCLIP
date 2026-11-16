import argparse
import csv
import json
from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from data_loader import create_dataloaders
from models.reliability_aware_model import ReliabilityAwareAVadCLIPModel
from models.visual_encoder import VisualEncoder


DEFAULT_CHECKPOINT = (
    "results/teacher_corruption_similarity_seed42_w0_05.pth"
)
CLASSIFICATION_THRESHOLD = 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest", default="dataset_manifest_675_test134.csv")
    parser.add_argument(
        "--audio-scale",
        type=float,
        default=1.0,
        help=(
            "Scale input audio features. This does not simulate waveform "
            "corruption; use --corruption-feature-dir for that."
        ),
    )
    parser.add_argument(
        "--corruption-feature-dir",
        help=(
            "Optional cache from prepare_corruption_dev_features.py. "
            "When set, evaluate clean audio and every waveform condition."
        ),
    )
    parser.add_argument("--reliability-output")
    args = parser.parse_args()

    if not 0.0 <= args.audio_scale <= 1.0:
        parser.error("--audio-scale must be between 0 and 1")
    checkpoint = Path(args.checkpoint)
    config_path = checkpoint.with_suffix(".config.json")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Checkpoint config not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.reliability_output:
        output_path = Path(args.reliability_output)
    elif args.corruption_feature_dir:
        output_path = checkpoint.with_name(
            f"{checkpoint.stem}_validation_reliability_by_condition.csv"
        )
    else:
        output_path = checkpoint.with_name(
            f"{checkpoint.stem}_validation_reliability_audio_scale_"
            f"{args.audio_scale:g}.csv"
        )
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite reliability log: {output_path}"
        )

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    _, val_loader = create_dataloaders(
        manifest_path=args.manifest,
        fold_index=None,
    )
    corruption_dir = (
        Path(args.corruption_feature_dir)
        if args.corruption_feature_dir
        else None
    )
    if corruption_dir is not None:
        if not corruption_dir.is_dir():
            raise FileNotFoundError(
                f"Development corruption cache not found: {corruption_dir}"
            )
        missing = [
            f"{Path(filename).stem}_corruptions.pt"
            for filename in val_loader.dataset.df["filename"]
            if not (
                corruption_dir
                / f"{Path(filename).stem}_corruptions.pt"
            ).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"Development corruption cache is incomplete "
                f"({len(missing)} files missing); prepare it with "
                "scripts/prepare_corruption_dev_features.py."
            )

    clip_encoder = VisualEncoder(device=device)
    reliability_config = config.get("reliability_estimator", {})
    detach_for_fusion = reliability_config.get(
        "fusion_gradient_policy",
        "",
    ).startswith("reliability is detached")
    model = ReliabilityAwareAVadCLIPModel(
        clip_model=clip_encoder.model,
        reliability_hidden_dim=reliability_config.get("hidden_dim", 64),
        detach_reliability_for_fusion=detach_for_fusion,
    ).to(device)
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    model.eval()

    predictions_by_condition = {}
    reliability_rows = []
    with torch.no_grad():
        for batch in val_loader:
            visual = batch["visual"].to(device)
            filename = batch["filename"][0]
            conditions = {"clean": batch["audio"].squeeze(0)}
            mute_masks = {}
            if corruption_dir is not None:
                cached = torch.load(
                    corruption_dir / f"{Path(filename).stem}_corruptions.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                conditions.update(
                    {
                        condition: variant["features"]
                        for condition, variant in cached.items()
                    }
                )
                if "partial_mute_50" in cached:
                    mute_masks["partial_mute_50"] = cached[
                        "partial_mute_50"
                    ]["mute_mask"].bool()

            label = int(batch["label"].item())
            for condition, condition_audio in conditions.items():
                audio = condition_audio.to(device) * args.audio_scale
                output = model(visual, audio.unsqueeze(0))
                score = float(output["video_score"].item())
                reliability = output["reliability_scores"].squeeze(0).cpu()
                audio_norm = output["audio_residual"].norm(
                    dim=-1
                ).squeeze(0).cpu()
                gated_norm = output["gated_audio_residual"].norm(
                    dim=-1
                ).squeeze(0).cpu()
                predicted = int(score >= CLASSIFICATION_THRESHOLD)
                row = {
                    "filename": filename,
                    "condition": condition,
                    "label": label,
                    "video_score": score,
                    "prediction_at_0_5": predicted,
                    "audio_scale": args.audio_scale,
                    "mean_segment_reliability": float(reliability.mean()),
                    "min_segment_reliability": float(reliability.min()),
                    "max_segment_reliability": float(reliability.max()),
                    "mean_audio_residual_norm": float(audio_norm.mean()),
                    "mean_gated_audio_residual_norm": float(gated_norm.mean()),
                    "mean_gated_to_ungated_norm_ratio": float(
                        (gated_norm / audio_norm.clamp_min(1e-8)).mean()
                    ),
                    "muted_segment_mean_reliability": "",
                    "retained_segment_mean_reliability": "",
                    "muted_segment_mean_gated_residual_norm": "",
                    "retained_segment_mean_gated_residual_norm": "",
                }
                if condition in mute_masks:
                    mute_mask = mute_masks[condition]
                    row.update(
                        {
                            "muted_segment_mean_reliability": float(
                                reliability[mute_mask].mean()
                            ),
                            "retained_segment_mean_reliability": float(
                                reliability[~mute_mask].mean()
                            ),
                            "muted_segment_mean_gated_residual_norm": float(
                                gated_norm[mute_mask].mean()
                            ),
                            "retained_segment_mean_gated_residual_norm": float(
                                gated_norm[~mute_mask].mean()
                            ),
                        }
                    )
                stats = predictions_by_condition.setdefault(
                    condition,
                    {"labels": [], "scores": [], "predictions": []},
                )
                stats["labels"].append(label)
                stats["scores"].append(score)
                stats["predictions"].append(predicted)
                row["segment_reliability_scores"] = json.dumps(
                    [float(value) for value in reliability]
                )
                reliability_rows.append(row)

    for condition, stats in predictions_by_condition.items():
        labels = stats["labels"]
        scores = stats["scores"]
        predicted_labels = stats["predictions"]
        metrics = {
            "accuracy": accuracy_score(labels, predicted_labels),
            "precision": precision_score(
                labels,
                predicted_labels,
                zero_division=0,
            ),
            "recall": recall_score(labels, predicted_labels, zero_division=0),
            "f1": f1_score(labels, predicted_labels, zero_division=0),
            "roc_auc": roc_auc_score(labels, scores),
        }
        rows = [
            row for row in reliability_rows
            if row["condition"] == condition
        ]
        mean_reliability = sum(
            row["mean_segment_reliability"] for row in rows
        ) / len(rows)
        mean_ratio = sum(
            row["mean_gated_to_ungated_norm_ratio"] for row in rows
        ) / len(rows)
        print(
            f"\n===== CORRUPTION-AWARE TEACHER VALIDATION "
            f"({condition}, audio scale {args.audio_scale:g}, "
            f"threshold 0.5) ====="
        )
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")
        print("Confusion Matrix:")
        print(confusion_matrix(labels, predicted_labels))
        print(
            "Mean reliability / gated-to-ungated residual ratio:",
            f"{mean_reliability:.4f} / {mean_ratio:.4f}",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=reliability_rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(reliability_rows)
    print(f"Per-video and per-segment reliability saved to {output_path}")


if __name__ == "__main__":
    main()
