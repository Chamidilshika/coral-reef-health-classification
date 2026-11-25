import sys
import json
import gc

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# Add project root to Python path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Project imports

from baseline_training.models import build_model
from baseline_training.config import (
    DEVICE,
    BATCH_SIZE,
    NUM_WORKERS
)

from preprocessing.transforms import eval_transform

from feature_extraction.config import (
    DATASETS,
    MODELS,
    SPLITS,
    EXPECTED_FEATURE_DIMENSIONS,
    EXPECTED_SPLIT_COUNTS,
    EXPERIMENTS_DIR,
    CHECKPOINT_DIR,
    OUTPUT_DIR
)

# Reproducibility

torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Load datasets

def create_feature_datasets(dataset_name):

    datasets = {}
    loaders = {}

    for split in SPLITS:

        split_dir = (
            EXPERIMENTS_DIR /
            dataset_name /
            split
        )

        if not split_dir.exists():
            raise FileNotFoundError(
                f"Dataset split not found: {split_dir}"
            )

        dataset = ImageFolder(
            root=str(split_dir),
            transform=eval_transform
        )

        expected_mapping = {
            "bleached": 0,
            "healthy": 1
        }

        if dataset.class_to_idx != expected_mapping:
            raise ValueError(
                f"Incorrect class mapping in {split_dir}. "
                f"Found: {dataset.class_to_idx}"
            )

        expected_count = EXPECTED_SPLIT_COUNTS[
            dataset_name
        ][split]

        if len(dataset) != expected_count:
            raise ValueError(
                f"Unexpected image count for "
                f"{dataset_name}/{split}. "
                f"Expected {expected_count}, "
                f"found {len(dataset)}."
            )

        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available()
        )

        datasets[split] = dataset
        loaders[split] = loader

        print(
            f"{dataset_name} | {split}: "
            f"{len(dataset)} images"
        )

    return datasets, loaders

# Load trained checkpoint

def load_trained_model(dataset_name, model_name):

    checkpoint_path = (
        CHECKPOINT_DIR /
        dataset_name /
        model_name /
        "best_model.pth"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Trained checkpoint not found: "
            f"{checkpoint_path}"
        )

    # Build the same architecture (used during training)
    model = build_model(model_name)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            f"model_state_dict not found in {checkpoint_path}"
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # Remove the classification head
    # The forward pass now returns the feature vector

    if model_name == "resnet50":

        model.fc = nn.Identity()

    elif model_name == "efficientnet_b0":

        model.classifier[1] = nn.Identity()

    elif model_name == "vit_small":

        model.head = nn.Identity()

    else:

        raise ValueError(
            f"Unsupported model: {model_name}"
        )

    model = model.to(DEVICE)
    model.eval()

    print(
        f"Loaded trained checkpoint: "
        f"{dataset_name}/{model_name}"
    )

    return model

# Extract features

def extract_features(model, loader, model_name):

    all_features = []
    all_labels = []

    model.eval()

    with torch.inference_mode():

        for images, labels in loader:

            images = images.to(DEVICE)

            features = model(images)

            # Ensure shape is [batch_size, feature_dim]
            features = features.flatten(start_dim=1)

            all_features.append(
                features.cpu().numpy()
            )

            all_labels.append(
                labels.numpy()
            )

    features_array = np.concatenate(
        all_features,
        axis=0
    ).astype(np.float32)

    labels_array = np.concatenate(
        all_labels,
        axis=0
    ).astype(np.int64)

    expected_dimension = (
        EXPECTED_FEATURE_DIMENSIONS[model_name]
    )

    if features_array.shape[1] != expected_dimension:
        raise ValueError(
            f"Incorrect feature dimension for {model_name}. "
            f"Expected {expected_dimension}, "
            f"found {features_array.shape[1]}."
        )

    if len(features_array) != len(labels_array):
        raise ValueError(
            "Feature and label counts do not match."
        )

    return features_array, labels_array

# Save extracted features

def save_features(
    dataset_name,
    model_name,
    split,
    features,
    labels
):

    model_output_dir = (
        OUTPUT_DIR /
        dataset_name /
        model_name
    )

    model_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        model_output_dir / f"{split}_features.npy",
        features
    )

    np.save(
        model_output_dir / f"{split}_labels.npy",
        labels
    )

    print(
        f"Saved {dataset_name}/{model_name}/{split}: "
        f"features={features.shape}, "
        f"labels={labels.shape}"
    )

# Process one dataset

def process_dataset(dataset_name):

    print("\n" + "=" * 65)
    print(f"FEATURE EXTRACTION: {dataset_name}")
    print("=" * 65)

    datasets, loaders = create_feature_datasets(
        dataset_name
    )

    # Store reference labels and paths

    reference_labels = {}
    reference_paths = {}

    for split in SPLITS:

        reference_labels[split] = np.asarray(
            datasets[split].targets,
            dtype=np.int64
        )

        reference_paths[split] = [
            str(Path(path).resolve())
            for path, _ in datasets[split].samples
        ]

    # Save a manifest of the image ordering

    manifest_dir = OUTPUT_DIR / dataset_name
    manifest_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    manifest = {}

    for split in SPLITS:

        manifest[split] = {
            "image_paths": reference_paths[split],
            "labels": reference_labels[split].tolist()
        }

    with open(
        manifest_dir / "sample_manifest.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            manifest,
            file,
            indent=4
        )

    # Extract features from all three models.

    for model_name in MODELS:

        print("\n" + "-" * 55)
        print(f"MODEL: {model_name}")
        print("-" * 55)

        model = load_trained_model(
            dataset_name,
            model_name
        )

        for split in SPLITS:

            features, labels = extract_features(
                model,
                loaders[split],
                model_name
            )

            # Verify that labels match the original dataset
            if not np.array_equal(
                labels,
                reference_labels[split]
            ):
                raise ValueError(
                    f"Label mismatch: "
                    f"{dataset_name}/{model_name}/{split}"
                )

            # Verify sample count.
            if len(features) != len(
                datasets[split]
            ):
                raise ValueError(
                    f"Sample count mismatch: "
                    f"{dataset_name}/{model_name}/{split}"
                )

            save_features(
                dataset_name,
                model_name,
                split,
                features,
                labels
            )

        del model
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(
        f"\nCompleted feature extraction for "
        f"{dataset_name}."
    )

# Main

def main():

    import argparse

    parser = argparse.ArgumentParser(
        description="Extract features from trained coral classification models."
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=DATASETS + ["all"]
    )

    args = parser.parse_args()

    selected_datasets = (
        DATASETS
        if args.dataset == "all"
        else [args.dataset]
    )

    for dataset_name in selected_datasets:

        process_dataset(dataset_name)

    print("\n" + "=" * 65)
    print("ALL REQUESTED FEATURE EXTRACTION COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()