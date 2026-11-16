import pandas as pd

df = pd.read_csv("dataset_manifest_675_test134.csv")

splits = {
    split: set(df[df["split"] == split]["source_movie"])
    for split in ("train", "val", "test")
}

for split, movies in splits.items():
    print(f"Unique movies in {split}:", len(movies))

overlaps = []
split_names = tuple(splits)
for index, first in enumerate(split_names):
    for second in split_names[index + 1:]:
        shared = splits[first].intersection(splits[second])
        overlaps.extend((first, second, movie) for movie in sorted(shared))

print("\nCross-split movie overlaps:", len(overlaps))
if overlaps:
    for first, second, movie in overlaps:
        print(f"{movie}: {first} / {second}")
else:
    print("No source-movie overlap detected!")
print("\nClass distribution:")

for split in ("train", "val", "test"):
    subset = df[df["split"] == split]

    print(f"\n{split.upper()}:")
    print(subset["binary_label"].value_counts().sort_index())

    print("Normal:", (subset["binary_label"] == 0).sum())
    print("Anomalous:", (subset["binary_label"] == 1).sum())