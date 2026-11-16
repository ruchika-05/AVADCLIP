
import torch
import csv
from pathlib import Path

from models.avadclip_model import AVadCLIPModel
from data_loader import create_dataloaders


def main():

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    _, val_loader = create_dataloaders()

    model = AVadCLIPModel().to(device)

    model.load_state_dict(
        torch.load(
            "best_avadclip_model.pth",
            map_location=device,
            weights_only=True
        )
    )

    model.eval()

    results = []

    with torch.no_grad():

        for batch in val_loader:

            audio = batch["audio"].to(device)
            visual = batch["visual"].to(device)

            output = model(visual, audio)

            score = output["video_score"].item()

            actual = int(batch["label"].item())
            predicted = int(score >= 0.5)

            filename = batch["filename"][0]
            category = batch["category"][0]

            result = {
                "filename": filename,
                "category": category,
                "actual": actual,
                "predicted": predicted,
                "score": round(score, 4),
                "correct": actual == predicted
            }

            results.append(result)

    print("\n===== VIDEO-WISE PREDICTIONS =====\n")

    for item in results:

        status = "CORRECT" if item["correct"] else "WRONG"

        print(f"Video: {item['filename']}")
        print(f"Category: {item['category']}")
        print(f"Actual: {item['actual']}")
        print(f"Predicted: {item['predicted']}")
        print(f"Anomaly score: {item['score']}")
        print(f"Result: {status}")
        print("-" * 50)

    # Save results to CSV
    Path("results").mkdir(exist_ok=True)

    with open(
        "results/validation_predictions.csv",
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys()
        )

        writer.writeheader()
        writer.writerows(results)

    print("\nResults saved to results/validation_predictions.csv")


if __name__ == "__main__":
    main()
