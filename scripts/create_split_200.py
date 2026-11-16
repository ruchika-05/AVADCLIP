
import pandas as pd
import random
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

OLD_SPLIT = Path("dataset_manifest_split.csv")
NEW_MANIFEST = Path("dataset_manifest_200.csv")
OUTPUT = Path("dataset_manifest_200_split.csv")

RANDOM_SEED = 42

old_df = pd.read_csv(OLD_SPLIT)
new_df = pd.read_csv(NEW_MANIFEST)

# Recover the original movie-level assignments
movie_splits = {}

for _, row in old_df.iterrows():
    movie = row["source_movie"]
    split = row["split"]

    if movie in movie_splits and movie_splits[movie] != split:
        raise ValueError(f"Conflicting old splits for {movie}")

    movie_splits[movie] = split

# Extract source movie names for all expanded videos
new_df["source_movie"] = new_df["filename"].apply(
    lambda name: Path(name).stem.split("__")[0]
)

# Preserve known movie assignments
new_df["split"] = new_df["source_movie"].map(movie_splits)

# Separate entirely new movies
unseen_mask = new_df["split"].isna()
unseen_df = new_df.loc[unseen_mask].copy()

if not unseen_df.empty:
    groups = unseen_df["source_movie"]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_SEED
    )

    train_idx, val_idx = next(
        splitter.split(unseen_df, groups=groups)
    )

    new_df.loc[unseen_df.index[train_idx], "split"] = "train"
    new_df.loc[unseen_df.index[val_idx], "split"] = "val"

# Verify no movie crosses splits
split_counts = new_df.groupby("source_movie")["split"].nunique()

if (split_counts > 1).any():
    raise ValueError("Source movie leakage detected!")

new_df.to_csv(OUTPUT, index=False)

print("Expanded split manifest created:", OUTPUT)
print("\nVideo counts:")
print(new_df["split"].value_counts())

print("\nUnique source movies:")
print(new_df.groupby("split")["source_movie"].nunique())

print("\nMovie leakage:", (split_counts > 1).sum())
print("Total videos:", len(new_df))
