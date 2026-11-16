
from data_loader import create_dataloaders

train_loader, val_loader = create_dataloaders()

print("Training videos:", len(train_loader))
print("Validation videos:", len(val_loader))

train_batch = next(iter(train_loader))
val_batch = next(iter(val_loader))

print("\nTraining batch:")
print("Audio:", train_batch["audio"].shape)
print("Visual:", train_batch["visual"].shape)
print("Label:", train_batch["label"])
print("Category:", train_batch["category"])
print("Category labels:", train_batch["category_labels"])

print("\nValidation batch:")
print("Audio:", val_batch["audio"].shape)
print("Visual:", val_batch["visual"].shape)
print("Label:", val_batch["label"])

print("\nDataLoader test completed!")
