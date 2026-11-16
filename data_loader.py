
import pandas as pd
import torch

from pathlib import Path
from torch.utils.data import Dataset, DataLoader


class VideoAnomalyDataset(Dataset):

    def __init__(self, manifest_path, feature_dir):

        self.df = pd.read_csv(manifest_path)
        self.feature_dir = Path(feature_dir)

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
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        stem = Path(row["filename"]).stem

        feature_path = (
            self.feature_dir / f"{stem}_multimodal.pt"
        )

        data = torch.load(
            feature_path,
            map_location="cpu",
            weights_only=True
        )

        # Audio and visual features
        audio = data["audio"].float()
        visual = data["visual"].float()

        # Binary label: 0 = Normal, 1 = Anomalous
        binary_label = torch.tensor(
            int(row["binary_label"]),
            dtype=torch.float32
        )

        # Multi-label category vector
        category_labels = torch.zeros(
            len(self.class_names),
            dtype=torch.float32
        )

        categories = str(row["categories"]).split(";")

        for category in categories:

            category = category.strip()

            if category in self.class_to_idx:
                idx = self.class_to_idx[category]
                category_labels[idx] = 1.0

        return {
            "audio": audio,
            "visual": visual,
            "label": binary_label,
            "category_labels": category_labels,
            "category": row["primary_category"],
            "filename": row["filename"]
        }


def create_dataloaders(
    manifest_path="dataset_manifest_675_test134.csv",
    fold_index=None,
):

    train_dataset = VideoAnomalyDataset(
        manifest_path,
        "features/multimodal"
    )

    val_dataset = VideoAnomalyDataset(
        manifest_path,
        "features/multimodal"
    )

    if fold_index is None:
        if "split" not in train_dataset.df.columns:
            raise ValueError(
                f"Manifest has no 'split' column: {manifest_path}"
            )
        train_mask = train_dataset.df["split"] == "train"
        val_mask = train_dataset.df["split"] == "val"
    else:
        if "fold" not in train_dataset.df.columns:
            raise ValueError(
                f"Manifest has no 'fold' column: {manifest_path}"
            )
        available_folds = set(train_dataset.df["fold"].astype(int).unique())
        if fold_index not in available_folds:
            raise ValueError(
                f"Fold {fold_index} is not in {manifest_path}; "
                f"available folds: {sorted(available_folds)}"
            )
        train_mask = train_dataset.df["fold"] != fold_index
        val_mask = train_dataset.df["fold"] == fold_index

    train_dataset.df = train_dataset.df[train_mask].reset_index(drop=True)

    val_dataset.df = val_dataset.df[val_mask].reset_index(drop=True)

    if train_dataset.df.empty or val_dataset.df.empty:
        raise ValueError(
            f"Empty train or validation split for fold {fold_index} "
            f"in {manifest_path}"
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False
    )

    return train_loader, val_loader
