
import pandas as pd
import torch
from pathlib import Path

df = pd.read_csv("dataset_manifest_675_test134.csv")

feature_dir = Path("features/multimodal")

lengths = []
missing = []
errors = []

for _, row in df.iterrows():
    stem = Path(row["filename"]).stem
    feature_path = feature_dir / f"{stem}_multimodal.pt"

    if not feature_path.exists():
        missing.append(row["filename"])
        continue

    try:
        data = torch.load(
            feature_path,
            map_location="cpu",
            weights_only=False
        )

        audio = data["audio"]
        visual = data["visual"]

        if audio.shape != visual.shape:
            errors.append((row["filename"], "Shape mismatch"))
            continue

        if audio.ndim != 2 or audio.shape[1] != 512:
            errors.append((row["filename"], "Unexpected feature shape"))
            continue

        if int(data["label"]) != int(row["binary_label"]):
            errors.append((row["filename"], "Label mismatch"))
            continue

        lengths.append(audio.shape[0])

    except Exception as e:
        errors.append((row["filename"], str(e)))

print("Total videos:", len(df))
print("Successfully checked:", len(lengths))
print("Missing files:", len(missing))
print("Errors:", len(errors))

if lengths:
    print("\nSegment statistics:")
    print("Minimum:", min(lengths))
    print("Maximum:", max(lengths))
    print("Average:", sum(lengths) / len(lengths))

print("\nCategories:")
print(df["primary_category"].value_counts())
