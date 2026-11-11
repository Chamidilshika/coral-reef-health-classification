import sys
from pathlib import Path
from torchvision import datasets

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from preprocessing.transforms import train_transform, eval_transform

EXPERIMENTS_DIR = PROJECT_DIR / "dataset" / "experiments"

experiments = [
    "kaggle_only",
    "self_collected_only",
    "combined"
]

for experiment in experiments:
    print(f"\n========== {experiment.upper()} ==========")

    dataset_dir = EXPERIMENTS_DIR / experiment

    train_data = datasets.ImageFolder(
        dataset_dir / "train",
        transform=train_transform
    )

    val_data = datasets.ImageFolder(
        dataset_dir / "val",
        transform=eval_transform
    )

    test_data = datasets.ImageFolder(
        dataset_dir / "test",
        transform=eval_transform
    )

    print("Classes:", train_data.classes)
    print("Training images:", len(train_data))
    print("Validation images:", len(val_data))
    print("Testing images:", len(test_data))

    image, label = train_data[0]

    print("Image shape:", image.shape)
    print("Sample label:", train_data.classes[label])

print("\nAll preprocessing tests passed!")