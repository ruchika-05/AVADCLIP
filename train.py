
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from models.avadclip_model import AVadCLIPModel
from models.alignment_loss import MILAlignLoss
from models.fusion import AudioVisualFusion
from data_loader import create_dataloaders
from models.visual_encoder import VisualEncoder


def apply_audio_dropout(audio, probability):
    if not 0.0 <= probability <= 1.0:
        raise ValueError("audio dropout probability must be between 0 and 1.")
    if audio.ndim < 2:
        raise ValueError("audio features must include batch and feature dimensions.")
    if probability == 0.0:
        return audio
    if probability == 1.0:
        return torch.zeros_like(audio)

    keep_mask = torch.rand(audio.shape[0], device=audio.device) >= probability
    mask_shape = (audio.shape[0],) + (1,) * (audio.ndim - 1)
    return audio * keep_mask.reshape(mask_shape).to(dtype=audio.dtype)


def apply_audio_degradation(audio, probability):
    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            "audio degradation probability must be between 0 and 1."
        )
    if audio.ndim < 2:
        raise ValueError("audio features must include batch and feature dimensions.")
    if probability == 0.0:
        return audio

    degrade_mask = torch.rand(audio.shape[0], device=audio.device) < probability
    scale_choices = torch.arange(
        5,
        device=audio.device,
        dtype=audio.dtype,
    ) / 4
    scales = scale_choices[
        torch.randint(0, len(scale_choices), (audio.shape[0],), device=audio.device)
    ]
    scales = torch.where(degrade_mask, scales, torch.ones_like(scales))
    mask_shape = (audio.shape[0],) + (1,) * (audio.ndim - 1)
    return audio * scales.reshape(mask_shape)


def training_audio_norm_reference(dataset):
    mean_norms = []
    for index in range(len(dataset)):
        audio = dataset[index]["audio"]
        mean_norms.append(torch.linalg.vector_norm(audio, dim=-1).mean())
    if not mean_norms:
        raise ValueError("Cannot calculate audio norm reference from empty dataset.")
    return float(torch.stack(mean_norms).median().item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--fusion-architecture",
        choices=AudioVisualFusion.ARCHITECTURES,
        default="joint",
        help="Fusion design: existing joint residual or separate modality branches.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional random seed for repeatable training runs.",
    )
    parser.add_argument(
        "--audio-dropout-probability",
        type=float,
        default=0.0,
        help=(
            "Probability of muting all audio features for a training video. "
            "When enabled, saves to a separate checkpoint by default."
        ),
    )
    parser.add_argument(
        "--audio-degradation-probability",
        type=float,
        default=0.0,
        help=(
            "Probability of randomly scaling a training video's audio by "
            "0, 0.25, 0.5, 0.75, or 1."
        ),
    )
    parser.add_argument(
        "--reliability-aware-audio",
        action="store_true",
        help=(
            "Use audio feature strength to reduce audio's contribution "
            "when the input audio is weak."
        ),
    )
    parser.add_argument(
        "--manifest",
        default="dataset_manifest_675_test134.csv",
    )
    parser.add_argument("--fold-index", type=int)
    parser.add_argument(
        "--checkpoint",
        help=(
            "Checkpoint path. Audio-augmentation runs use a separate path "
            "by default."
        ),
    )
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    if not 0.0 <= args.audio_dropout_probability <= 1.0:
        parser.error("--audio-dropout-probability must be between 0 and 1")
    if not 0.0 <= args.audio_degradation_probability <= 1.0:
        parser.error("--audio-degradation-probability must be between 0 and 1")
    if (
        args.reliability_aware_audio
        and args.audio_degradation_probability == 0.0
        and args.audio_dropout_probability == 0.0
    ):
        parser.error(
            "--reliability-aware-audio requires audio degradation or dropout "
            "during training"
        )
    if args.fusion_architecture == "separate" and (
        args.reliability_aware_audio
        or args.audio_dropout_probability > 0.0
        or args.audio_degradation_probability > 0.0
    ):
        parser.error(
            "The separate-branch experiment must use unaugmented audio "
            "without reliability gating."
        )

    checkpoint = args.checkpoint
    if checkpoint is None:
        if args.fusion_architecture == "separate":
            checkpoint = (
                f"results/teacher_separate_branch_fold_{args.fold_index}.pth"
                if args.fold_index is not None
                else "results/teacher_separate_branch.pth"
            )
        elif (
            args.audio_dropout_probability > 0.0
            or args.audio_degradation_probability > 0.0
            or args.reliability_aware_audio
        ):
            checkpoint = (
                f"results/teacher_reliability_fold_{args.fold_index}.pth"
                if args.fold_index is not None
                else "results/teacher_reliability.pth"
            )
        else:
            checkpoint = (
                f"results/teacher_fold_{args.fold_index}.pth"
                if args.fold_index is not None
                else "best_avadclip_teacher.pth"
            )

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    train_loader, val_loader = create_dataloaders(
        manifest_path=args.manifest,
        fold_index=args.fold_index,
    )
    config_path = Path(checkpoint).with_suffix(".config.json")
    if args.fusion_architecture == "separate" and (
        Path(checkpoint).exists() or config_path.exists()
    ):
        parser.error(
            "Refusing to overwrite an existing separate-branch checkpoint "
            f"or config: {checkpoint}"
        )
    audio_norm_reference = (
        training_audio_norm_reference(train_loader.dataset)
        if args.reliability_aware_audio
        else None
    )
    config_path.parent.mkdir(parents=True, exist_ok=True)
    training_config = {
        "model": "AVadCLIP audio-visual teacher",
        "fusion_architecture": args.fusion_architecture,
        "manifest": args.manifest,
        "fold_index": args.fold_index,
        "checkpoint": checkpoint,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "audio_dropout_probability": args.audio_dropout_probability,
        "audio_dropout": (
            "randomly zero all audio features for selected training videos"
        ),
        "audio_degradation_probability": (
            args.audio_degradation_probability
        ),
        "audio_degradation_scales": [0.0, 0.25, 0.5, 0.75, 1.0],
        "reliability_aware_audio": args.reliability_aware_audio,
        "audio_norm_reference": audio_norm_reference,
        "optimizer": "Adam",
        "batch_size": train_loader.batch_size,
        "train_videos": len(train_loader.dataset),
        "validation_videos": len(val_loader.dataset),
        "device": (
            "mps" if torch.backends.mps.is_available() else "cpu"
        ),
        "loss": "binary cross-entropy + MIL alignment loss",
        "alignment_logit_temperature": 0.07,
        "video_top_k": "min(int(sequence_length / 16 + 1), sequence_length)",
        "visual_max_length": 768,
        "visual_transformer_layers": 2,
        "visual_attention_heads": 1,
        "visual_attention_window": 4,
        "similarity_graph": (
            "learned shared linear projection followed by cosine similarity, "
            "threshold 0.7, softmax over rows"
        ),
        "distance_graph": "exp(-absolute_temporal_distance / e)",
        "audio_tcn": {
            "levels": 3,
            "kernel_size": 2,
            "dilations": [1, 2, 4],
            "dropout": 0.1,
        },
        "binary_anomaly_branch": (
            "Linear(512, 2048), QuickGELU, Linear(2048, 512), "
            "residual addition, Linear(512, 1), Sigmoid"
        ),
        "validation_checkpoint_selection": args.fold_index is None,
    }
    with config_path.open("w", encoding="utf-8") as config_file:
        json.dump(training_config, config_file, indent=2)
    print(f"Training configuration saved to {config_path}")

    clip_encoder = VisualEncoder(device=device)
    model = AVadCLIPModel(
        clip_model=clip_encoder.model,
        audio_norm_reference=audio_norm_reference,
        fusion_architecture=args.fusion_architecture,
    ).to(device)

    binary_criterion = nn.BCELoss()
    alignment_criterion = MILAlignLoss(num_classes=7).to(device)

    optimizer = optim.Adam(
        list(model.parameters()) + list(alignment_criterion.parameters()),
        lr=args.learning_rate,
    )

    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_binary_loss = 0.0
        train_alignment_loss = 0.0

        for batch in train_loader:
            audio = batch["audio"].to(device)
            audio = apply_audio_dropout(
                audio,
                args.audio_dropout_probability,
            )
            audio = apply_audio_degradation(
                audio,
                args.audio_degradation_probability,
            )
            visual = batch["visual"].to(device)
            labels = batch["label"].to(device)
            category_labels = batch["category_labels"].to(device)

            optimizer.zero_grad()
            output = model(visual, audio)
            binary_loss = binary_criterion(output["video_score"], labels)
            alignment = alignment_criterion(
                output["alignment_map"],
                category_labels,
                lengths=[output["segment_scores"].shape[1]],
            )
            alignment_loss = alignment["alignment_loss"]
            loss = binary_loss + alignment_loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_binary_loss += binary_loss.item()
            train_alignment_loss += alignment_loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                audio = batch["audio"].to(device)
                visual = batch["visual"].to(device)
                labels = batch["label"].to(device)
                category_labels = batch["category_labels"].to(device)

                output = model(visual, audio)
                binary_loss = binary_criterion(output["video_score"], labels)
                alignment = alignment_criterion(
                    output["alignment_map"],
                    category_labels,
                    lengths=[output["segment_scores"].shape[1]],
                )
                val_loss += (binary_loss + alignment["alignment_loss"]).item()

        train_loss /= len(train_loader)
        train_binary_loss /= len(train_loader)
        train_alignment_loss /= len(train_loader)
        val_loss /= len(val_loader)

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"(binary: {train_binary_loss:.4f}, "
            f"alignment: {train_alignment_loss:.4f}) "
            f"Val Loss: {val_loss:.4f}"
        )

        if args.fold_index is not None:
            if epoch == args.epochs - 1:
                torch.save(model.state_dict(), checkpoint)
                print(f"Final fold checkpoint saved to {checkpoint}")
        elif val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint)
            print(f"Best teacher checkpoint saved to {checkpoint}")

    print("\nTraining completed!")
    if args.fold_index is None:
        training_config["best_validation_loss"] = best_val_loss
        with config_path.open("w", encoding="utf-8") as config_file:
            json.dump(training_config, config_file, indent=2)
        print("Best validation loss:", best_val_loss)
    else:
        print("Fold validation loss is reported for analysis only.")


if __name__ == "__main__":
    main()
