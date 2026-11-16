import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from models.avadclip_model import AVadCLIPModel
from models.avadclip_student import AVadCLIPStudent
from models.visual_encoder import VisualEncoder


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "results/final_test_evaluation_config.json"
OUTPUT_PATHS = (
    ROOT / "results/final_test_results.json",
    ROOT / "results/final_test_metrics.csv",
    ROOT / "results/final_test_confusion_matrices.png",
    ROOT / "results/final_test_predictions.csv",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class UnlabeledTestDataset(Dataset):
    def __init__(self, manifest_path, feature_dir):
        self.manifest = pd.read_csv(
            manifest_path,
            usecols=["filename", "split"],
        )
        self.manifest = self.manifest[
            self.manifest["split"] == "test"
        ].reset_index(drop=True)
        self.feature_dir = Path(feature_dir)
        if self.manifest.empty:
            raise ValueError("No test videos found in the manifest.")
        if self.manifest["filename"].duplicated().any():
            raise ValueError("Test manifest contains duplicate filenames.")

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, index):
        filename = str(self.manifest.iloc[index]["filename"])
        feature_path = (
            self.feature_dir
            / f"{Path(filename).stem}_multimodal.pt"
        )
        data = torch.load(
            feature_path,
            map_location="cpu",
            weights_only=True,
        )
        audio = data["audio"].float()
        visual = data["visual"].float()
        if (
            audio.ndim != 2
            or visual.ndim != 2
            or audio.shape[1] != 512
            or visual.shape[1] != 512
        ):
            raise ValueError(
                f"Invalid feature shape for {filename}: "
                f"audio={tuple(audio.shape)}, visual={tuple(visual.shape)}"
            )
        if audio.shape[0] != visual.shape[0]:
            raise ValueError(f"Audio/visual length mismatch for {filename}")
        if not torch.isfinite(audio).all() or not torch.isfinite(visual).all():
            raise ValueError(f"Non-finite features found for {filename}")
        if data.get("filename") != Path(filename).stem:
            raise ValueError(f"Feature/video identifier mismatch: {filename}")

        return {
            "filename": filename,
            "audio": audio,
            "visual": visual,
        }


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()


def load_frozen_model(model_name, clip_model, checkpoint_path, device):
    if model_name == "teacher":
        model = AVadCLIPModel(clip_model=clip_model).to(device)
    elif model_name == "student":
        model = AVadCLIPStudent(clip_model=clip_model).to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def safe_ratio(numerator, denominator):
    if denominator == 0:
        return None
    return numerator / denominator


def calculate_metrics(labels, scores, predictions):
    tn = int(((labels == 0) & (predictions == 0)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    tp = int(((labels == 1) & (predictions == 1)).sum())
    precision = safe_ratio(tp, tp + fp)
    recall = safe_ratio(tp, tp + fn)
    f1 = safe_ratio(2 * tp, 2 * tp + fp + fn)
    try:
        auc = float(roc_auc_score(labels, scores))
    except ValueError:
        auc = None

    return {
        "accuracy": safe_ratio(tn + tp, len(labels)),
        "anomaly_precision": precision,
        "anomaly_recall": recall,
        "anomaly_f1": f1,
        "roc_auc": auc,
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
        "correctly_detected_anomalies": tp,
        "missed_anomalies": fn,
        "false_positive_normal_videos": fp,
    }


def main():
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(
            f"Frozen evaluation config is missing: {CONFIG_PATH}"
        )
    existing_outputs = [path for path in OUTPUT_PATHS if path.exists()]
    if existing_outputs:
        raise FileExistsError(
            "Refusing to overwrite final evaluation outputs: "
            + ", ".join(str(path) for path in existing_outputs)
        )

    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = json.load(file)
    if config.get("evaluation_status") != "frozen_before_test_inference":
        raise ValueError("Evaluation decisions are not marked frozen.")
    if config.get("test_labels_read_before_freeze") is not False:
        raise ValueError("Config does not certify labels were unread at freeze.")
    if config["evaluation_protocol"].get(
        "thresholds_frozen_before_test_inference"
    ) is not True:
        raise ValueError("Classification thresholds were not frozen.")

    manifest_path = ROOT / config["manifest"]
    if sha256_file(manifest_path) != config["manifest_sha256"]:
        raise ValueError("Manifest changed after the evaluation was frozen.")
    model_settings = config["model_decisions"]
    for name in ("teacher", "student"):
        settings = model_settings[name]
        checkpoint_path = ROOT / settings["checkpoint"]
        if sha256_file(checkpoint_path) != settings["checkpoint_sha256"]:
            raise ValueError(f"{name} checkpoint changed after freeze.")
        dev_path = ROOT / settings["development_config"]
        with dev_path.open(encoding="utf-8") as file:
            development_config = json.load(file)
        saved_threshold = development_config["validation_metrics"]["threshold"]
        if saved_threshold != settings["classification_threshold"]:
            raise ValueError(
                f"{name} threshold does not match the saved development result."
            )
        if settings["threshold_source"] != "saved development config":
            raise ValueError(f"{name} threshold source is not documented.")

    dataset = UnlabeledTestDataset(
        manifest_path,
        ROOT / config["feature_pipeline"]["feature_directory"],
    )
    if len(dataset) != config["split_sizes"]["reserved_test"]:
        raise ValueError(
            f"Expected {config['split_sizes']['reserved_test']} test videos, "
            f"found {len(dataset)}."
        )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    clip_encoder = VisualEncoder(device=device)
    teacher = load_frozen_model(
        "teacher",
        clip_encoder.model,
        ROOT / model_settings["teacher"]["checkpoint"],
        device,
    )
    student = load_frozen_model(
        "student",
        clip_encoder.model,
        ROOT / model_settings["student"]["checkpoint"],
        device,
    )

    predictions = []
    elapsed = {"teacher": 0.0, "student": 0.0}
    print(
        f"Running frozen inference on {len(dataset)} test videos "
        f"using {device}; labels are not included in the inference dataset."
    )
    with torch.inference_mode():
        for index, batch in enumerate(loader, 1):
            visual = batch["visual"].to(device)
            audio = batch["audio"].to(device)
            filename = batch["filename"][0]

            synchronize(device)
            start = time.perf_counter()
            teacher_output = teacher(visual, audio)
            synchronize(device)
            elapsed["teacher"] += time.perf_counter() - start

            synchronize(device)
            start = time.perf_counter()
            student_output = student(visual)
            synchronize(device)
            elapsed["student"] += time.perf_counter() - start

            teacher_score = float(teacher_output["video_score"].item())
            student_score = float(student_output["video_score"].item())
            if not math.isfinite(teacher_score) or not math.isfinite(student_score):
                raise ValueError(f"Non-finite anomaly score for {filename}")
            predictions.append(
                {
                    "filename": filename,
                    "teacher_anomaly_score": teacher_score,
                    "student_anomaly_score": student_score,
                }
            )
            if index % 20 == 0 or index == len(dataset):
                print(f"Inferred {index}/{len(dataset)}")

    del teacher, student
    score_frame = pd.DataFrame(predictions)
    if score_frame["filename"].tolist() != dataset.manifest["filename"].tolist():
        raise RuntimeError("Inference output order differs from test manifest.")

    # Test labels are first read only after both frozen models have inferred
    # every test video and the unlabeled scores are held separately.
    labels_frame = pd.read_csv(
        manifest_path,
        usecols=["filename", "split", "binary_label"],
    )
    labels_frame = labels_frame[labels_frame["split"] == "test"][
        ["filename", "binary_label"]
    ].reset_index(drop=True)
    if labels_frame["filename"].tolist() != score_frame["filename"].tolist():
        raise RuntimeError("Test labels do not align with inference order.")

    labels = labels_frame["binary_label"].astype(int).to_numpy()
    if not set(labels).issubset({0, 1}):
        raise ValueError("Test labels must be binary 0/1.")
    total_normals = int((labels == 0).sum())
    total_anomalies = int((labels == 1).sum())
    if total_normals + total_anomalies != len(dataset):
        raise RuntimeError("Test label count does not equal inference count.")

    score_frame["ground_truth_label"] = labels
    results = {
        "evaluation_type": "final independent reserved test evaluation",
        "test_labels_used_during_inference": False,
        "test_data_used_for_training_or_tuning": False,
        "manifest": config["manifest"],
        "total_test_videos": len(dataset),
        "normal_test_videos": total_normals,
        "anomalous_test_videos": total_anomalies,
        "thresholds": {
            name: model_settings[name]["classification_threshold"]
            for name in ("teacher", "student")
        },
        "models": {},
        "inference_time_seconds": elapsed,
        "inference_time_method": config["evaluation_protocol"]["inference_time"],
    }
    for name in ("teacher", "student"):
        score_column = f"{name}_anomaly_score"
        prediction_column = f"{name}_predicted_label"
        threshold = model_settings[name]["classification_threshold"]
        scores = score_frame[score_column].to_numpy(dtype=float)
        predicted = (scores >= threshold).astype(int)
        score_frame[prediction_column] = predicted
        results["models"][name] = {
            "checkpoint": model_settings[name]["checkpoint"],
            "threshold": threshold,
            **calculate_metrics(labels, scores, predicted),
        }

    metric_rows = []
    for name in ("teacher", "student"):
        model_result = results["models"][name]
        confusion = model_result["confusion_matrix"]
        metric_rows.append(
            {
                "model": name,
                "accuracy": model_result["accuracy"],
                "anomaly_precision": model_result["anomaly_precision"],
                "anomaly_recall": model_result["anomaly_recall"],
                "anomaly_f1": model_result["anomaly_f1"],
                "roc_auc": model_result["roc_auc"],
                "tn": confusion["tn"],
                "fp": confusion["fp"],
                "fn": confusion["fn"],
                "tp": confusion["tp"],
                "correctly_detected_anomalies": (
                    model_result["correctly_detected_anomalies"]
                ),
                "missed_anomalies": model_result["missed_anomalies"],
                "false_positive_normal_videos": (
                    model_result["false_positive_normal_videos"]
                ),
                "threshold": model_result["threshold"],
                "total_test_videos": len(dataset),
                "normal_test_videos": total_normals,
                "anomalous_test_videos": total_anomalies,
                "inference_time_seconds": elapsed[name],
            }
        )

    predictions_path = ROOT / "results/final_test_predictions.csv"
    score_frame = score_frame[
        [
            "filename",
            "ground_truth_label",
            "teacher_anomaly_score",
            "teacher_predicted_label",
            "student_anomaly_score",
            "student_predicted_label",
        ]
    ]
    score_frame.to_csv(predictions_path, index=False)
    pd.DataFrame(metric_rows).to_csv(
        ROOT / "results/final_test_metrics.csv",
        index=False,
    )
    with (ROOT / "results/final_test_results.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(results, file, indent=2, allow_nan=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, name in zip(axes, ("teacher", "student")):
        confusion = results["models"][name]["confusion_matrix"]
        matrix = [
            [confusion["tn"], confusion["fp"]],
            [confusion["fn"], confusion["tp"]],
        ]
        display = ConfusionMatrixDisplay(
            confusion_matrix=np.asarray(matrix, dtype=int),
            display_labels=["Normal", "Anomalous"],
        )
        display.plot(ax=axis, cmap="Blues", colorbar=False, values_format="d")
        axis.set_title(f"{name.title()} (threshold {results['thresholds'][name]:.2f})")
    figure.tight_layout()
    figure.savefig(
        ROOT / "results/final_test_confusion_matrices.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)

    config["evaluation_status"] = "completed"
    config["test_labels_read_before_freeze"] = False
    config["test_inference_completed_before_labels_read"] = True
    config["results"] = results
    with CONFIG_PATH.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, allow_nan=False)

    for name in ("teacher", "student"):
        model_result = results["models"][name]
        print(
            f"{name.title()}: accuracy={model_result['accuracy']:.4f}, "
            f"precision={model_result['anomaly_precision']}, "
            f"recall={model_result['anomaly_recall']}, "
            f"F1={model_result['anomaly_f1']}, "
            f"ROC-AUC={model_result['roc_auc']}"
        )
        print("Confusion matrix TN/FP/FN/TP:", model_result["confusion_matrix"])
    print("Saved final test evaluation artifacts in results/.")


if __name__ == "__main__":
    main()
