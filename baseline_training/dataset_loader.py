
import sys
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# Add project root to Python import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    EXPERIMENTS_DIR,
    CLASS_TO_IDX,
    BATCH_SIZE,
    NUM_WORKERS
)

from preprocessing.transforms import (
    train_transform,
    eval_transform
)


def create_dataloaders(dataset_name):

    dataset_path = EXPERIMENTS_DIR / dataset_name

    train_dataset = ImageFolder(
        dataset_path / "train",
        transform=train_transform
    )

    val_dataset = ImageFolder(
        dataset_path / "val",
        transform=eval_transform
    )

    test_dataset = ImageFolder(
        dataset_path / "test",
        transform=eval_transform
    )

    # Confirm class mapping
    for name, dataset in [
        ("Training", train_dataset),
        ("Validation", val_dataset),
        ("Testing", test_dataset)
    ]:
        if dataset.class_to_idx != CLASS_TO_IDX:
            raise ValueError(
                f"{name} class mapping is incorrect: "
                f"{dataset.class_to_idx}"
            )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(__import__("torch").cuda.is_available())
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(__import__("torch").cuda.is_available())
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(__import__("torch").cuda.is_available())
    )

    print(f"\nDataset: {dataset_name}")
    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    print(f"Testing images: {len(test_dataset)}")
    print(f"Class mapping: {train_dataset.class_to_idx}")

    return train_loader, val_loader, test_loader