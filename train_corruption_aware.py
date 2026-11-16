import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

from data_loader import create_dataloaders
from models.audio_corruption import CORRUPTION_CONDITIONS
from models.alignment_loss import MILAlignLoss
from models.reliability_aware_model import ReliabilityAwareAVadCLIPModel
from models.reliability_estimator import AudioReliabilityEstimator
from models.visual_encoder import VisualEncoder


CORRUPTED_CONDITIONS = CORRUPTION_CONDITIONS
DEFAULT_CHECKPOINT = (
    "results/teacher_corruption_similarity_seed42_w0_05.pth"
)


def _atomic_torch_save(value, path):
    temporary_path = path.with_name(path.name + ".tmp")
    try:
        torch.save(value, temporary_path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _load_condition(dataset, corruption_dir, index, condition):
    sample = dataset[index]
    if condition == "clean":
        audio = sample["audio"]
        targets = torch.ones(audio.shape[0], dtype=torch.float32)
        return audio, targets

    stem = Path(sample["filename"]).stem
    path = corruption_dir / f"{stem}_corruptions.pt"
    variants = torch.load(path, map_location="cpu", weights_only=True)
    variant = variants[condition]
    return variant["features"].float(), variant["targets"].float()


def _sample_condition(rng):
    if rng.random() < 0.5:
        return "clean"
    return rng.choice(CORRUPTED_CONDITIONS)


def _reliability_validation(
    estimator,
    dataset,
    corruption_dir,
    indices,
    device,
):
    estimator.eval()
    condition_stats = {}
    all_targets = []
    all_predictions = []
    corrupted_predictions = []
    clean_reliabilities = []
    clean_vs_corrupted_labels = []
    clean_vs_corrupted_scores = []

    with torch.no_grad():
        for index in indices:
            clean_sample = dataset[index]
            clean_audio = clean_sample["audio"].to(device).unsqueeze(0)
            clean_prediction = estimator(clean_audio).squeeze(0).cpu()
            clean_target = torch.ones_like(clean_prediction)
            clean_reliabilities.extend(clean_prediction.tolist())
            all_targets.extend(clean_target.tolist())
            all_predictions.extend(clean_prediction.tolist())
            clean_vs_corrupted_labels.extend([0] * len(clean_prediction))
            clean_vs_corrupted_scores.extend(
                (1.0 - clean_prediction).tolist()
            )
            clean_stats = condition_stats.setdefault(
                "clean",
                {"targets": [], "predictions": []},
            )
            clean_stats["targets"].extend(clean_target.tolist())
            clean_stats["predictions"].extend(clean_prediction.tolist())

            stem = Path(clean_sample["filename"]).stem
            variants = torch.load(
                corruption_dir / f"{stem}_corruptions.pt",
                map_location="cpu",
                weights_only=True,
            )
            for condition, variant in variants.items():
                audio = variant["features"].float().to(device).unsqueeze(0)
                target = variant["targets"].float()
                prediction = estimator(audio).squeeze(0).cpu()
                all_targets.extend(target.tolist())
                all_predictions.extend(prediction.tolist())
                corrupted_predictions.extend(prediction.tolist())
                clean_vs_corrupted_labels.extend([1] * len(prediction))
                clean_vs_corrupted_scores.extend((1.0 - prediction).tolist())
                stats = condition_stats.setdefault(
                    condition,
                    {"targets": [], "predictions": []},
                )
                stats["targets"].extend(target.tolist())
                stats["predictions"].extend(prediction.tolist())
                if condition == "partial_mute_50":
                    mute_mask = variant.get(
                        "mute_mask",
                        variant["targets"] <= 0.01,
                    ).bool()
                    stats.setdefault("muted_predictions", []).extend(
                        prediction[mute_mask].tolist()
                    )
                    stats.setdefault("retained_predictions", []).extend(
                        prediction[~mute_mask].tolist()
                    )

    target_tensor = torch.tensor(all_targets)
    prediction_tensor = torch.tensor(all_predictions)
    constant_prediction = target_tensor.mean()
    mae = (prediction_tensor - target_tensor).abs().mean().item()
    constant_mae = (constant_prediction - target_tensor).abs().mean().item()
    clean_mean = float(np.mean(clean_reliabilities))
    corrupted_mean = float(np.mean(corrupted_predictions))
    output = {
        "heldout_train_videos": len(indices),
        "segment_mae": mae,
        "constant_baseline_mae": constant_mae,
        "mean_clean_reliability": clean_mean,
        "mean_corrupted_reliability": corrupted_mean,
        "clean_vs_corrupted_roc_auc": float(
            roc_auc_score(
                clean_vs_corrupted_labels,
                clean_vs_corrupted_scores,
            )
        ),
        "condition_means": {
            name: {
                "target": float(np.mean(values["targets"])),
                "prediction": float(np.mean(values["predictions"])),
                **(
                    {
                        "muted_prediction": float(
                            np.mean(values["muted_predictions"])
                        ),
                        "retained_prediction": float(
                            np.mean(values["retained_predictions"])
                        ),
                    }
                    if name == "partial_mute_50"
                    and values.get("muted_predictions")
                    and values.get("retained_predictions")
                    else {}
                ),
            }
            for name, values in condition_stats.items()
        },
        "lower_than_constant_baseline": mae < constant_mae,
    }
    return output


def _fusion_validation(model, dataset, corruption_dir, indices, device):
    model.eval()
    condition_stats = {}
    with torch.no_grad():
        for index in indices:
            sample = dataset[index]
            visual = sample["visual"].to(device).unsqueeze(0)
            stem = Path(sample["filename"]).stem
            clean_audio = sample["audio"]
            clean_target = torch.ones(clean_audio.shape[0])
            conditions = [("clean", clean_audio, clean_target)]

            variants = torch.load(
                corruption_dir / f"{stem}_corruptions.pt",
                map_location="cpu",
                weights_only=True,
            )
            conditions.extend(
                (
                    condition,
                    variant["features"].float(),
                    variant["targets"].float(),
                )
                for condition, variant in variants.items()
            )

            for condition, audio, targets in conditions:
                output = model(visual, audio.to(device).unsqueeze(0))
                reliability = output["reliability_scores"].squeeze(0).cpu()
                audio_norm = output["audio_residual"].norm(
                    dim=-1
                ).squeeze(0).cpu()
                gated_norm = output["gated_audio_residual"].norm(
                    dim=-1
                ).squeeze(0).cpu()
                stats = condition_stats.setdefault(
                    condition,
                    {
                        "reliability": [],
                        "audio_residual_norm": [],
                        "gated_residual_norm": [],
                        "gated_to_ungated_ratio": [],
                    },
                )
                stats["reliability"].extend(reliability.tolist())
                stats["audio_residual_norm"].extend(audio_norm.tolist())
                stats["gated_residual_norm"].extend(gated_norm.tolist())
                stats["gated_to_ungated_ratio"].extend(
                    (gated_norm / audio_norm.clamp_min(1e-8)).tolist()
                )
                if condition == "partial_mute_50":
                    mute_mask = variants[condition].get(
                        "mute_mask",
                        targets <= 0.01,
                    ).bool()
                    stats.setdefault("muted_reliability", []).extend(
                        reliability[mute_mask].tolist()
                    )
                    stats.setdefault("retained_reliability", []).extend(
                        reliability[~mute_mask].tolist()
                    )
                    stats.setdefault("muted_gated_norm", []).extend(
                        gated_norm[mute_mask].tolist()
                    )
                    stats.setdefault("retained_gated_norm", []).extend(
                        gated_norm[~mute_mask].tolist()
                    )

    report = {}
    for condition, values in condition_stats.items():
        report[condition] = {
            "mean_reliability": float(np.mean(values["reliability"])),
            "mean_audio_residual_norm": float(
                np.mean(values["audio_residual_norm"])
            ),
            "mean_gated_audio_residual_norm": float(
                np.mean(values["gated_residual_norm"])
            ),
            "mean_gated_to_ungated_norm_ratio": float(
                np.mean(values["gated_to_ungated_ratio"])
            ),
        }
        if condition == "partial_mute_50":
            report[condition].update(
                {
                    "muted_segment_mean_reliability": float(
                        np.mean(values["muted_reliability"])
                    ),
                    "retained_segment_mean_reliability": float(
                        np.mean(values["retained_reliability"])
                    ),
                    "muted_segment_mean_gated_residual_norm": float(
                        np.mean(values["muted_gated_norm"])
                    ),
                    "retained_segment_mean_gated_residual_norm": float(
                        np.mean(values["retained_gated_norm"])
                    ),
                }
            )
    return report


def _pretrain_estimator(
    estimator,
    dataset,
    corruption_dir,
    train_indices,
    validation_indices,
    device,
    epochs,
    learning_rate,
    seed,
):
    optimizer = optim.Adam(estimator.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    rng = random.Random(seed)
    for epoch in range(epochs):
        estimator.train()
        order = list(train_indices)
        rng.shuffle(order)
        total_loss = 0.0
        for index in order:
            condition = _sample_condition(rng)
            audio, targets = _load_condition(
                dataset,
                corruption_dir,
                index,
                condition,
            )
            audio = audio.to(device).unsqueeze(0)
            targets = targets.to(device).unsqueeze(0)
            optimizer.zero_grad()
            prediction = estimator(audio)
            loss = criterion(prediction, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(
            f"Reliability pretrain [{epoch + 1}/{epochs}] "
            f"loss: {total_loss / len(order):.4f}"
        )

    report = _reliability_validation(
        estimator,
        dataset,
        corruption_dir,
        validation_indices,
        device,
    )
    print("Held-out training-video reliability diagnostic:")
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train an isolated segment-level corruption-aware AVadCLIP teacher. "
            "All reliability labels and waveform corruptions come from train."
        )
    )
    parser.add_argument("--manifest", default="dataset_manifest_675_test134.csv")
    parser.add_argument("--fold-index", type=int)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--reliability-pretrain-epochs", type=int, default=5)
    parser.add_argument("--reliability-learning-rate", type=float, default=1e-3)
    parser.add_argument("--reliability-loss-weight", type=float, default=0.05)
    parser.add_argument("--reliability-hidden-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--corruption-feature-dir",
        default="features/corruption_aware_train_similarity",
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()

    if args.fold_index is not None:
        parser.error(
            "This experiment uses the existing train/validation split, not folds."
        )
    if args.epochs < 1 or args.reliability_pretrain_epochs < 1:
        parser.error("Training epoch counts must be positive.")
    if args.reliability_loss_weight < 0:
        parser.error("--reliability-loss-weight must be non-negative.")

    checkpoint = Path(args.checkpoint)
    config_path = checkpoint.with_suffix(".config.json")
    estimator_path = checkpoint.with_name(
        f"{checkpoint.stem}.reliability_estimator.pth"
    )
    if any(path.exists() for path in (checkpoint, config_path, estimator_path)):
        parser.error(
            "Refusing to overwrite an existing experiment artifact: "
            f"{checkpoint}, {config_path}, or {estimator_path}"
        )

    corruption_dir = Path(args.corruption_feature_dir)
    if not corruption_dir.is_dir():
        parser.error(
            "Missing training-only waveform-corruption cache. Run "
            "scripts/relabel_corruption_train_cache.py after preparing the "
            "training corruption features."
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print("Using device:", device)

    train_loader, val_loader = create_dataloaders(
        manifest_path=args.manifest,
        fold_index=None,
    )
    train_dataset = train_loader.dataset
    train_ids = list(range(len(train_dataset)))
    split_rng = random.Random(args.seed)
    split_rng.shuffle(train_ids)
    heldout_count = max(1, round(0.2 * len(train_ids)))
    validation_ids = train_ids[:heldout_count]
    estimator_train_ids = train_ids[heldout_count:]

    expected_cache_files = {
        Path(filename).stem + "_corruptions.pt"
        for filename in train_dataset.df["filename"]
    }
    missing_cache = sorted(
        name for name in expected_cache_files
        if not (corruption_dir / name).is_file()
    )
    if missing_cache:
        parser.error(
            f"Corruption cache is incomplete ({len(missing_cache)} missing); "
            "regenerate it with --resume."
        )

    estimator = AudioReliabilityEstimator(
        feature_dim=512,
        hidden_dim=args.reliability_hidden_dim,
    ).to(device)
    reliability_report = _pretrain_estimator(
        estimator,
        train_dataset,
        corruption_dir,
        estimator_train_ids,
        validation_ids,
        device,
        args.reliability_pretrain_epochs,
        args.reliability_learning_rate,
        args.seed,
    )

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "model": "AVadCLIP corruption-aware audio-visual teacher",
        "fusion_architecture": "visual residual + reliability-gated audio residual",
        "fusion_equation": "V + G_v(V) + r_t * G_a(A)",
        "reliability_estimator": {
            "input": "raw Wav2CLIP feature for each aligned segment",
            "architecture": f"Linear(512, {args.reliability_hidden_dim}), GELU, Linear({args.reliability_hidden_dim}, 1), Sigmoid",
            "hidden_dim": args.reliability_hidden_dim,
            "supervision": (
                "regression to per-segment cosine similarity between clean "
                "and corrupted Wav2CLIP embeddings, clamped to [0, 1]; "
                "muted and silent segments target 0"
            ),
            "fusion_gradient_policy": (
                "reliability is detached before fusion, so anomaly and "
                "alignment losses cannot train the estimator"
            ),
            "heldout_policy": (
                "held-out training videos are excluded from both estimator "
                "pretraining and reliability loss during teacher training"
            ),
            "pretrain_epochs": args.reliability_pretrain_epochs,
            "learning_rate": args.reliability_learning_rate,
            "loss": "mean squared error",
            "train_video_count": len(estimator_train_ids),
            "heldout_train_video_count": len(validation_ids),
            "heldout_diagnostic": reliability_report,
        },
        "corruption_conditions": {
            "volume_0_5": {"waveform_scale": 0.5},
            "volume_0_1": {"waveform_scale": 0.1},
            "noise_snr_20": {"snr_db": 20},
            "noise_snr_10": {"snr_db": 10},
            "noise_snr_0": {"snr_db": 0},
            "partial_mute_50": {
                "definition": "deterministic contiguous half of 1-second segments muted",
                "target_reliability": (
                    "0 for muted segments; cosine similarity to clean "
                    "embedding for retained segments"
                ),
            },
            "silence": {
                "waveform_scale": 0.0,
                "target_reliability": 0.0,
            },
            "clean": {"target_reliability": 1.0},
        },
        "manifest": args.manifest,
        "fold_index": None,
        "training_corruption_feature_dir": str(corruption_dir),
        "checkpoint": str(checkpoint),
        "reliability_estimator_checkpoint": str(estimator_path),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "reliability_loss_weight": args.reliability_loss_weight,
        "optimizer": "Adam",
        "batch_size": train_loader.batch_size,
        "train_videos": len(train_dataset),
        "validation_videos": len(val_loader.dataset),
        "device": str(device),
        "anomaly_loss": "binary cross-entropy + MIL alignment loss",
        "validation_checkpoint_selection": "binary + MIL alignment loss; clean validation audio only",
        "evaluation_threshold": 0.5,
        "audio_condition_sampling": (
            "clean with probability 0.5; otherwise one corruption condition "
            "chosen uniformly"
        ),
        "reserved_test_accessed": False,
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Training configuration saved to {config_path}")

    clip_encoder = VisualEncoder(device=device)
    model = ReliabilityAwareAVadCLIPModel(
        clip_model=clip_encoder.model,
        reliability_hidden_dim=args.reliability_hidden_dim,
        detach_reliability_for_fusion=True,
    ).to(device)
    model.reliability_estimator.load_state_dict(estimator.state_dict())

    binary_criterion = nn.BCELoss()
    reliability_criterion = nn.MSELoss()
    alignment_criterion = MILAlignLoss(num_classes=7).to(device)
    optimizer = optim.Adam(
        list(model.parameters()) + list(alignment_criterion.parameters()),
        lr=args.learning_rate,
    )
    best_val_loss = float("inf")
    train_indices_by_filename = {
        filename: index
        for index, filename in enumerate(train_dataset.df["filename"])
    }
    reliability_train_ids = set(estimator_train_ids)

    for epoch in range(args.epochs):
        model.train()
        train_totals = {"total": 0.0, "binary": 0.0, "alignment": 0.0, "reliability": 0.0}
        for batch in train_loader:
            condition = _sample_condition(random)
            index = train_indices_by_filename[batch["filename"][0]]
            audio, reliability_targets = _load_condition(
                train_dataset,
                corruption_dir,
                int(index),
                condition,
            )
            audio = audio.unsqueeze(0).to(device)
            reliability_targets = reliability_targets.unsqueeze(0).to(device)
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
            reliability_loss = (
                reliability_criterion(
                    output["reliability_scores"],
                    reliability_targets,
                )
                if int(index) in reliability_train_ids
                else torch.zeros((), device=device)
            )
            loss = (
                binary_loss
                + alignment_loss
                + args.reliability_loss_weight * reliability_loss
            )
            loss.backward()
            optimizer.step()

            train_totals["total"] += loss.item()
            train_totals["binary"] += binary_loss.item()
            train_totals["alignment"] += alignment_loss.item()
            train_totals["reliability"] += reliability_loss.item()

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

        count = len(train_loader)
        val_loss /= len(val_loader)
        print(
            f"Epoch [{epoch + 1}/{args.epochs}] "
            f"Train Loss: {train_totals['total'] / count:.4f} "
            f"(binary: {train_totals['binary'] / count:.4f}, "
            f"alignment: {train_totals['alignment'] / count:.4f}, "
            f"reliability: {train_totals['reliability'] / count:.4f}) "
            f"Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _atomic_torch_save(model.state_dict(), checkpoint)
            _atomic_torch_save(
                model.reliability_estimator.state_dict(),
                estimator_path,
            )
            print(f"Best teacher checkpoint saved to {checkpoint}")

    config["best_validation_loss"] = best_val_loss
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    final_reliability_report = _reliability_validation(
        model.reliability_estimator,
        train_dataset,
        corruption_dir,
        validation_ids,
        device,
    )
    config["reliability_estimator"]["heldout_diagnostic"] = (
        final_reliability_report
    )
    fusion_report = _fusion_validation(
        model,
        train_dataset,
        corruption_dir,
        validation_ids,
        device,
    )
    config["heldout_train_fusion_diagnostic"] = fusion_report
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("Held-out training-video gated-residual diagnostic:")
    print(json.dumps(fusion_report, indent=2))
    print("\nTraining completed!")
    print("Best validation loss:", best_val_loss)


if __name__ == "__main__":
    main()
