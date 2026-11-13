
import os
import sys
import argparse
import numpy as np
import torch

# Get the main project directory
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# Add the project directory to Python's import path
sys.path.insert(0, PROJECT_ROOT)

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from models import load_all_models
from preprocessing.transforms import eval_transform

def extract_split_features(
    split_name,
    dataset_path,
    output_path,
    models_dict,
    device,
    batch_size=16
):

    # Load images using the evaluation transformation
    dataset = ImageFolder(
        root=dataset_path,
        transform=eval_transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    print(f"\nProcessing {split_name} split")
    print("Images:", len(dataset))
    print("Classes:", dataset.class_to_idx)

    # Store features separately for each model
    features_dict = {
        model_name: []
        for model_name in models_dict
    }

    all_labels = []
    all_paths = []

    # Feature extraction (no model training)
    with torch.no_grad():

        for batch_index, (images, labels) in enumerate(dataloader):

            images = images.to(device)

            for model_name, model in models_dict.items():

                features = model(images)

                features_dict[model_name].append(
                    features.cpu().numpy()
                )

            all_labels.append(labels.numpy())    

            start = batch_index * batch_size
            end = start + len(labels)

            batch_paths = [
                dataset.samples[i][0]
                for i in range(start, end)
            ]

            all_paths.extend(batch_paths)

            print(
                f"Processed {min(end, len(dataset))}"
                f" / {len(dataset)} images",
                end="\r"
            )

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    labels_array = np.concatenate(all_labels)
    paths_array = np.array(all_paths)

    # Save the extracted features for each model
    for model_name, feature_batches in features_dict.items():

        feature_array = np.concatenate(
            feature_batches,
            axis=0
        )

        save_file = os.path.join(
            output_path,
            f"{model_name}_{split_name}.npz"
        )

        np.savez_compressed(
            save_file,
            features=feature_array,
            labels=labels_array,
            paths=paths_array,
            class_names=np.array(dataset.classes)
        )

        print(
            f"\nSaved {model_name}: "
            f"{feature_array.shape}"
        )

    print(f"Completed {split_name} split!")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="kaggle_only",
        choices=[
            "kaggle_only",
            "self_collected_only",
            "combined"
        ]
    )

    args = parser.parse_args()

    project_root = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    dataset_root = os.path.join(
        project_root,
        "dataset",
        "experiments",
        args.dataset
    )

    output_root = os.path.join(
        project_root,
        "extracted_features",
        args.dataset
    )

    models_dict, device = load_all_models()

    print("Using device:", device)
    print("Dataset:", args.dataset)

    for split_name in ["train", "val", "test"]:

        split_path = os.path.join(
            dataset_root,
            split_name
        )

        if not os.path.isdir(split_path):
            raise FileNotFoundError(
                f"Dataset split not found: {split_path}"
            )

        extract_split_features(
            split_name=split_name,
            dataset_path=split_path,
            output_path=output_root,
            models_dict=models_dict,
            device=device,
            batch_size=16
        )

    print("\nAll feature extraction completed!")


if __name__ == "__main__":
    main()
         