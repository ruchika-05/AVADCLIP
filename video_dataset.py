
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class VideoAnomalyDataset(Dataset):

    def __init__(self, manifest_path="dataset_manifest.csv"):

        self.manifest = pd.read_csv(manifest_path)

        self.feature_dir = Path("features/multimodal")

        # All categories present in our dataset
        self.class_names = [
            "Normal",
            "Fighting",
            "Shooting",
            "Explosion",
            "Car accident",
            "Abuse",
            "Riot"
        ]

        self.class_to_idx = {
            name: idx
            for idx, name in enumerate(self.class_names)
        }

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, index):

        row = self.manifest.iloc[index]

        filename = Path(row["filename"]).stem

        feature_path = (
            self.feature_dir /
            f"{filename}_multimodal.pt"
        )

        data = torch.load(
            feature_path,
            map_location="cpu",
            weights_only=True
        )

        audio = data["audio"].float()
        visual = data["visual"].float()

        # Binary anomaly label
        binary_label = torch.tensor(
            data["label"],
            dtype=torch.float32
        )

        # Multi-label category vector
        category_labels = torch.zeros(
            len(self.class_names),
            dtype=torch.float32
        )

        for category in data["categories"]:

            if category in self.class_to_idx:

                idx = self.class_to_idx[category]
                category_labels[idx] = 1.0

        return {
            "audio": audio,
            "visual": visual,
            "label": binary_label,
            "category_labels": category_labels,
            "filename": filename
        }


if __name__ == "__main__":

    dataset = VideoAnomalyDataset()

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True
    )

    print("Total videos:", len(dataset))
    print("Categories:", dataset.class_names)

    # Test one batch
    batch = next(iter(loader))

    print("\nDataset loading successful!")
    print("Audio shape:", batch["audio"].shape)
    print("Visual shape:", batch["visual"].shape)
    print("Binary label:", batch["label"])
    print("Category labels:", batch["category_labels"])
    print("Filename:", batch["filename"])
