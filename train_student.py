import argparse
import json
import math
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from data_loader import create_dataloaders
from models.alignment_loss import MILAlignLoss
from models.avadclip_model import AVadCLIPModel
from models.avadclip_student import (
    AVadCLIPStudent,
    normal_anomaly_prompt_regularization,
)
from models.uncertainty_distillation import (
    uncertainty_weighted_distillation_loss,
)
from models.visual_encoder import VisualEncoder


def binary_classification_loss(scores, labels, positive_class_weight=1.0):
    if positive_class_weight <= 0:
        raise ValueError("positive_class_weight must be greater than zero.")

    per_video_loss = nn.functional.binary_cross_entropy(
        scores,
        labels,
        reduction="none",
    )
    class_weights = torch.where(
        labels == 1,
        positive_class_weight,
        1.0,
    )
    return (per_video_loss * class_weights).mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--distillation-weight", type=float, default=1e-8)
    parser.add_argument("--positive-class-weight", type=float, default=1.0)
    parser.add_argument("--prompt-regularization-weight", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--scheduler-milestones",
        type=int,
        nargs="+",
        default=[3, 6, 10],
    )
    parser.add_argument("--scheduler-rate", type=float, default=0.1)
    parser.add_argument("--manifest", default="dataset_manifest_675_test134.csv")
    parser.add_argument("--fold-index", type=int)
    parser.add_argument("--teacher-checkpoint")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    if args.positive_class_weight <= 0:
        parser.error("--positive-class-weight must be greater than zero")

    teacher_checkpoint = args.teacher_checkpoint
    checkpoint = args.checkpoint
    if teacher_checkpoint is None:
        teacher_checkpoint = (
            f"results/teacher_fold_{args.fold_index}.pth"
            if args.fold_index is not None
            else "best_avadclip_teacher.pth"
        )
    if checkpoint is None:
        checkpoint = (
            f"results/student_fold_{args.fold_index}.pth"
            if args.fold_index is not None
            else "best_avadclip_student.pth"
        )

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; this training step expects MPS.")
    device = torch.device("mps")
    torch.manual_seed(args.seed)

    train_loader, val_loader = create_dataloaders(
        manifest_path=args.manifest,
        fold_index=args.fold_index,
    )
    config_path = Path(checkpoint).with_suffix(".config.json")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    training_config = {
        "model": "AVadCLIP visual-only student",
        "manifest": args.manifest,
        "fold_index": args.fold_index,
        "teacher_checkpoint": teacher_checkpoint,
        "checkpoint": checkpoint,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "optimizer": "AdamW",
        "scheduler": {
            "type": "MultiStepLR",
            "milestones": args.scheduler_milestones,
            "gamma": args.scheduler_rate,
        },
        "distillation_weight": args.distillation_weight,
        "positive_class_weight": args.positive_class_weight,
        "binary_training_loss": (
            "binary cross entropy with the configured positive-class "
            "multiplier"
        ),
        "prompt_regularization_weight": (
            args.prompt_regularization_weight
        ),
        "checkpoint_selection_metric": (
            "validation binary loss + alignment loss; "
            "distillation loss excluded"
        ),
        "distillation_weight_note": (
            "This zero-weight run is a no-distillation control."
            if args.distillation_weight == 0
            else (
                "The official XD-Violence student training script uses "
                "1e-8; compare loss and gradient scales before changing it."
                if args.distillation_weight == 1e-8
                else (
                    "This active coefficient differs from the official "
                    "repository's 1e-8."
                )
            )
        ),
        "student_text_conditioning": (
            "anomaly-logit-weighted visual context plus residual "
            "four-times-width QuickGELU text MLP"
        ),
        "student_visual_transformer": {
            "layers": 1,
            "attention_window": 64,
        },
        "prompt_regularization": (
            "mean absolute cosine similarity between normal and anomalous "
            "conditioned class features"
        ),
        "batch_size": train_loader.batch_size,
        "train_videos": len(train_loader.dataset),
        "validation_videos": len(val_loader.dataset),
        "device": str(device),
        "visual_max_length": 768,
        "similarity_graph": (
            "learned shared linear projection followed by cosine "
            "similarity, threshold 0.7, softmax over rows"
        ),
        "distance_graph": "exp(-absolute_temporal_distance / e)",
        "distillation_objective": (
            "mean((teacher - student)^2 / "
            "(1e-6 + 2 * exp(log_variance)) + log_variance)"
        ),
        "alignment_logit_temperature": 0.07,
        "preflight": {"status": "pending"},
    }
    with config_path.open("w", encoding="utf-8") as config_file:
        json.dump(training_config, config_file, indent=2)
    print(f"Training configuration saved to {config_path}")

    clip_encoder = VisualEncoder(device=device)

    teacher = AVadCLIPModel(clip_model=clip_encoder.model).to(device)
    teacher.load_state_dict(
        torch.load(
            teacher_checkpoint,
            map_location=device,
            weights_only=True,
        )
    )
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    if any(parameter.requires_grad for parameter in teacher.parameters()):
        raise RuntimeError("Teacher parameters must be frozen before student training.")

    student = AVadCLIPStudent(clip_model=clip_encoder.model).to(device)
    student.text_prompt.context.data.copy_(
        teacher.text_prompt.context.detach()
    )

    preflight_batch = next(iter(train_loader))
    preflight_visual = preflight_batch["visual"].to(device)
    preflight_audio = preflight_batch["audio"].to(device)
    if preflight_visual.shape[1] != preflight_audio.shape[1]:
        raise ValueError(
            "Teacher audio and visual sequence lengths differ: "
            f"{preflight_audio.shape[1]} vs {preflight_visual.shape[1]}"
        )

    with torch.no_grad():
        preflight_teacher_output = teacher(
            preflight_visual,
            preflight_audio,
        )
    preflight_student_output = student(preflight_visual)
    teacher_features = preflight_teacher_output["fused_features"]
    student_features = preflight_student_output["student_features"]
    log_variance = preflight_student_output["log_variance"]
    if teacher_features.shape != student_features.shape:
        raise ValueError(
            "Teacher/student feature shapes do not match: "
            f"{tuple(teacher_features.shape)} vs "
            f"{tuple(student_features.shape)}"
        )
    if student_features.shape != log_variance.shape:
        raise ValueError(
            "Student feature/log-variance shapes do not match: "
            f"{tuple(student_features.shape)} vs {tuple(log_variance.shape)}"
        )

    raw_preflight_loss = uncertainty_weighted_distillation_loss(
        teacher_features,
        student_features,
        log_variance,
    )
    weighted_preflight_loss = (
        args.distillation_weight * raw_preflight_loss
    )
    checked_parameters = [
        student.feature_estimator.feature_enhancement
        .bottlenecks[0].conv[0].weight,
        student.feature_estimator.uncertainty_head[0].weight,
    ]
    preflight_gradients = torch.autograd.grad(
        weighted_preflight_loss,
        checked_parameters,
        allow_unused=True,
    )
    gradient_norms = [
        None if gradient is None else gradient.norm().item()
        for gradient in preflight_gradients
    ]
    if any(
        norm is None or not math.isfinite(norm)
        for norm in gradient_norms
    ):
        raise RuntimeError(
            "Weighted distillation loss produced missing or non-finite "
            f"student gradients: {gradient_norms}"
        )
    if args.distillation_weight > 0 and any(
        norm <= 0 for norm in gradient_norms
    ):
        raise RuntimeError(
            "Weighted distillation loss did not produce finite, nonzero "
            f"gradients for student feature and uncertainty parameters: "
            f"{gradient_norms}"
        )
    if args.distillation_weight == 0 and any(
        norm != 0 for norm in gradient_norms
    ):
        raise RuntimeError(
            "Zero-weight distillation unexpectedly produced nonzero "
            f"student gradients: {gradient_norms}"
        )

    training_config["preflight"] = {
        "status": "passed",
        "teacher_checkpoint_loaded_strictly": True,
        "teacher_parameters_frozen": True,
        "student_forward_input": "visual features only",
        "teacher_student_feature_shape": list(teacher_features.shape),
        "audio_visual_temporal_lengths_match": True,
        "distillation_weight": args.distillation_weight,
        "raw_distillation_loss": raw_preflight_loss.item(),
        "weighted_distillation_loss": weighted_preflight_loss.item(),
        "student_gradient_norms": gradient_norms,
        "gradient_expectation": (
            "zero because distillation weight is 0"
            if args.distillation_weight == 0
            else "finite and nonzero"
        ),
        "optimizer_step_performed": False,
    }
    with config_path.open("w", encoding="utf-8") as config_file:
        json.dump(training_config, config_file, indent=2)
    print(
        "Preflight passed: teacher loaded/frozen, visual-only student, "
        f"matched feature shape {tuple(teacher_features.shape)}, "
        f"raw distillation loss {raw_preflight_loss.item():.6f}, "
        f"weighted loss {weighted_preflight_loss.item():.6f}, "
        f"student gradient norms {gradient_norms}"
    )

    alignment_criterion = MILAlignLoss(num_classes=7).to(device)
    binary_criterion = nn.BCELoss()
    optimizer = optim.AdamW(
        list(student.parameters()) + list(alignment_criterion.parameters()),
        lr=args.learning_rate,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=args.scheduler_milestones,
        gamma=args.scheduler_rate,
    )

    best_val_task_loss = float("inf")
    best_val_epoch = None
    for epoch in range(args.epochs):
        student.train()
        train_loss = 0.0
        train_binary_loss = 0.0
        train_alignment_loss = 0.0
        train_prompt_regularization_loss = 0.0
        train_distillation_loss = 0.0
        train_weighted_distillation_loss = 0.0

        for batch in train_loader:
            audio = batch["audio"].to(device)
            visual = batch["visual"].to(device)
            labels = batch["label"].to(device)
            category_labels = batch["category_labels"].to(device)

            with torch.no_grad():
                teacher_output = teacher(visual, audio)

            student_output = student(visual)
            binary_loss = binary_classification_loss(
                student_output["video_score"],
                labels,
                positive_class_weight=args.positive_class_weight,
            )
            alignment = alignment_criterion(
                student_output["alignment_map"],
                category_labels,
                lengths=[student_output["segment_scores"].shape[1]],
            )
            alignment_loss = alignment["alignment_loss"]
            prompt_regularization_loss = (
                normal_anomaly_prompt_regularization(
                    student_output["conditioned_class_features"]
                )
            )
            weighted_prompt_regularization_loss = (
                args.prompt_regularization_weight
                * prompt_regularization_loss
            )
            distillation_loss = uncertainty_weighted_distillation_loss(
                teacher_output["fused_features"],
                student_output["student_features"],
                student_output["log_variance"],
            )
            weighted_distillation_loss = (
                args.distillation_weight * distillation_loss
            )
            loss = (
                binary_loss
                + alignment_loss
                + weighted_distillation_loss
                + weighted_prompt_regularization_loss
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_binary_loss += binary_loss.item()
            train_alignment_loss += alignment_loss.item()
            train_prompt_regularization_loss += (
                prompt_regularization_loss.item()
            )
            train_distillation_loss += distillation_loss.item()
            train_weighted_distillation_loss += (
                weighted_distillation_loss.item()
            )

        student.eval()
        val_loss = 0.0
        val_binary_loss = 0.0
        val_alignment_loss = 0.0
        val_prompt_regularization_loss = 0.0
        val_distillation_loss = 0.0
        val_weighted_distillation_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                audio = batch["audio"].to(device)
                visual = batch["visual"].to(device)
                labels = batch["label"].to(device)
                category_labels = batch["category_labels"].to(device)

                teacher_output = teacher(visual, audio)
                student_output = student(visual)
                binary_loss = binary_criterion(
                    student_output["video_score"],
                    labels,
                )
                alignment = alignment_criterion(
                    student_output["alignment_map"],
                    category_labels,
                    lengths=[student_output["segment_scores"].shape[1]],
                )
                prompt_regularization_loss = (
                    normal_anomaly_prompt_regularization(
                        student_output["conditioned_class_features"]
                    )
                )
                distillation_loss = uncertainty_weighted_distillation_loss(
                    teacher_output["fused_features"],
                    student_output["student_features"],
                    student_output["log_variance"],
                )
                weighted_distillation_loss = (
                    args.distillation_weight * distillation_loss
                )
                val_loss += (
                    binary_loss
                    + alignment["alignment_loss"]
                    + weighted_distillation_loss
                    + args.prompt_regularization_weight
                    * prompt_regularization_loss
                ).item()
                val_binary_loss += binary_loss.item()
                val_alignment_loss += alignment["alignment_loss"].item()
                val_prompt_regularization_loss += (
                    prompt_regularization_loss.item()
                )
                val_distillation_loss += distillation_loss.item()
                val_weighted_distillation_loss += (
                    weighted_distillation_loss.item()
                )

        train_size = len(train_loader)
        val_size = len(val_loader)
        train_loss /= train_size
        train_binary_loss /= train_size
        train_alignment_loss /= train_size
        train_prompt_regularization_loss /= train_size
        train_distillation_loss /= train_size
        train_weighted_distillation_loss /= train_size
        val_loss /= val_size
        val_binary_loss /= val_size
        val_alignment_loss /= val_size
        val_task_loss = val_binary_loss + val_alignment_loss
        val_prompt_regularization_loss /= val_size
        val_distillation_loss /= val_size
        val_weighted_distillation_loss /= val_size

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"(binary: {train_binary_loss:.4f}, "
            f"alignment: {train_alignment_loss:.4f}, "
            f"prompt reg: {train_prompt_regularization_loss:.4f}, "
            f"distillation raw: {train_distillation_loss:.4f}, "
            f"weighted: {train_weighted_distillation_loss:.4f}) "
            f"Val Loss: {val_loss:.4f} "
            f"(checkpoint task loss: {val_task_loss:.4f}, "
            f"binary: {val_binary_loss:.4f}, "
            f"alignment: {val_alignment_loss:.4f}) "
            f"(prompt reg: {val_prompt_regularization_loss:.4f}) "
            f"(distillation raw: {val_distillation_loss:.4f}, "
            f"weighted: {val_weighted_distillation_loss:.4f})"
        )
        scheduler.step()

        if args.fold_index is not None:
            if epoch == args.epochs - 1:
                torch.save(student.state_dict(), checkpoint)
                print(f"Final fold checkpoint saved to {checkpoint}")
        elif val_task_loss < best_val_task_loss:
            best_val_task_loss = val_task_loss
            best_val_epoch = epoch + 1
            torch.save(student.state_dict(), checkpoint)
            print(
                "Best student checkpoint saved to "
                f"{checkpoint} (validation task loss "
                f"{best_val_task_loss:.4f})"
            )

    print("\nStudent training completed!")
    if args.fold_index is None:
        training_config["best_validation_loss"] = best_val_task_loss
        training_config["best_validation_epoch"] = best_val_epoch
        training_config["final_learning_rate"] = optimizer.param_groups[0]["lr"]
        training_config["training_completed"] = True
        with config_path.open("w", encoding="utf-8") as config_file:
            json.dump(training_config, config_file, indent=2)
        print("Best validation task loss:", best_val_task_loss)
    else:
        training_config["training_completed"] = True
        with config_path.open("w", encoding="utf-8") as config_file:
            json.dump(training_config, config_file, indent=2)
        print("Fold validation loss is reported for analysis only.")


if __name__ == "__main__":
    main()
