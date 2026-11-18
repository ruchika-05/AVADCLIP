
import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

from models.avadclip_model import AVadCLIPModel
from models.avadclip_student import AVadCLIPStudent
from data_loader import create_dataloaders
from models.visual_encoder import VisualEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=("teacher", "student"),
        default="teacher",
    )
    parser.add_argument("--checkpoint")
    parser.add_argument("--manifest", default="dataset_manifest_675_test134.csv")
    parser.add_argument("--fold-index", type=int)
    parser.add_argument(
        "--corruption-feature-dir",
        help=(
            "Optional development-only waveform-corruption feature cache. "
            "For teachers, reports metrics separately for clean and each "
            "cached waveform condition."
        ),
    )
    parser.add_argument(
        "--audio-scale",
        type=float,
        default=1.0,
        help="Scale validation audio features from 0 (muted) to 1 (unchanged).",
    )
    args = parser.parse_args()

    if not 0.0 <= args.audio_scale <= 1.0:
        parser.error("--audio-scale must be between 0 and 1")
    if args.model == "student" and args.audio_scale != 1.0:
        parser.error("--audio-scale applies only to the audio-visual teacher")
    if args.model == "student" and args.corruption_feature_dir:
        parser.error("--corruption-feature-dir applies only to the teacher")

    checkpoint = args.checkpoint
    if checkpoint is None and args.fold_index is not None:
        checkpoint = (
            f"results/teacher_fold_{args.fold_index}.pth"
            if args.model == "teacher"
            else f"results/student_fold_{args.fold_index}.pth"
        )

    # Select device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Load validation data
    _, val_loader = create_dataloaders(
        manifest_path=args.manifest,
        fold_index=args.fold_index,
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
                f"({len(missing)} files missing)."
            )

    clip_encoder = VisualEncoder(device=device)
    if args.model == "teacher":
        checkpoint = checkpoint or "best_avadclip_teacher.pth"
        config_path = Path(checkpoint).with_suffix(".config.json")
        audio_norm_reference = None
        fusion_architecture = "joint"
        if config_path.is_file():
            with config_path.open(encoding="utf-8") as file:
                training_config = json.load(file)
            audio_norm_reference = training_config.get("audio_norm_reference")
            fusion_architecture = training_config.get(
                "fusion_architecture",
                "joint",
            )
        if fusion_architecture == "separate" and audio_norm_reference is not None:
            raise ValueError(
                "Separate-branch checkpoint config must not specify audio "
                "reliability."
            )
        model = AVadCLIPModel(
            clip_model=clip_encoder.model,
            audio_norm_reference=audio_norm_reference,
            fusion_architecture=fusion_architecture,
        ).to(device)
    else:
        model = AVadCLIPStudent(clip_model=clip_encoder.model).to(device)
        checkpoint = checkpoint or "best_avadclip_student.pth"

    model.load_state_dict(
        torch.load(
            checkpoint,
            map_location=device,
            weights_only=True
        )
    )

    model.eval()

    predictions_by_condition = {}

    with torch.no_grad():
        for batch in val_loader:
            visual = batch["visual"].to(device)
            filename = batch["filename"][0]
            conditions = {"clean": batch["audio"].squeeze(0)}
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
            label = int(batch["label"].item())
            for condition, condition_audio in conditions.items():
                audio = condition_audio.to(
                    device=device,
                    dtype=next(model.audio_temporal.parameters()).dtype
                ) * args.audio_scale
                if args.model == "teacher":
                    output = model(visual, audio.unsqueeze(0))
                else:
                    output = model(visual)
                score = float(output["video_score"].item())
                stats = predictions_by_condition.setdefault(
                    condition,
                    {"labels": [], "scores": [], "predictions": []},
                )
                stats["labels"].append(label)
                stats["scores"].append(score)
                stats["predictions"].append(int(score >= 0.5))

    for condition, stats in predictions_by_condition.items():
        actual_labels = stats["labels"]
        predicted_labels = stats["predictions"]
        predicted_scores = stats["scores"]
        accuracy = accuracy_score(actual_labels, predicted_labels)
        precision = precision_score(
            actual_labels,
            predicted_labels,
            zero_division=0,
        )
        recall = recall_score(
            actual_labels,
            predicted_labels,
            zero_division=0,
        )
        f1 = f1_score(
            actual_labels,
            predicted_labels,
            zero_division=0,
        )
        auc = roc_auc_score(actual_labels, predicted_scores)
        print(
            f"\n===== {args.model.upper()} VALIDATION "
            f"({condition}, audio scale {args.audio_scale:g}) RESULTS ====="
        )
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-score:  {f1:.4f}")
        print(f"ROC-AUC:   {auc:.4f}")
        print("Confusion Matrix:")
        print(confusion_matrix(actual_labels, predicted_labels))


if __name__ == "__main__":
    main()
