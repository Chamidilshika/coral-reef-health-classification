import os
import argparse
import numpy as np

MODEL_NAMES = [
    "resnet50",
    "efficientnet_b0",
    "vit_small"
]

EXPECTED_DIMENSIONS = {
    "resnet50": 2048,
    "efficientnet_b0": 1280,
    "vit_small": 384
}

DATASETS = [
    "kaggle_only",
    "self_collected_only",
    "combined"
]

SPLITS = ["train", "val", "test"]

def normalize_path(path):

    """Normalize a saved image path for reliable comparison."""
    if isinstance(path, bytes):
        path = path.decode("utf-8")

    return os.path.normcase(os.path.normpath(str(path)))

def fuse_split(project_root, dataset_name, split):

    input_folder = os.path.join(
        project_root,
        "extracted_features",
        dataset_name
    )

    output_folder = os.path.join(
        project_root,
        "fused_features",
        dataset_name
    )

    os.makedirs(output_folder, exist_ok=True)

    feature_arrays = []
    reference_labels = None
    reference_paths = None
    reference_class_names = None

    print(f"\nProcessing {dataset_name} - {split}")

    for model_name in MODEL_NAMES:

        filename = f"{model_name}_{split}.npz"
        file_path = os.path.join(input_folder, filename)

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Missing feature file: {file_path}"
            )

        with np.load(file_path) as data:

            features = data["features"]
            labels = data["labels"]
            paths = data["paths"]
            class_names = data["class_names"]

            expected_dimension = EXPECTED_DIMENSIONS[model_name]

            if features.ndim != 2:
                raise ValueError(
                    f"{filename}: Features must be a 2D array."
                )

            if features.shape[1] != expected_dimension:
                raise ValueError(
                    f"{filename}: Expected {expected_dimension} "
                    f"features, got {features.shape[1]}."
                )

            if not np.isfinite(features).all():
                raise ValueError(
                    f"{filename}: Features contain NaN or infinity."
                )

            if len(features) != len(labels) or len(features) != len(paths):
                raise ValueError(
                    f"{filename}: Feature, label, and path counts differ."
                )

            normalized_paths = [
                normalize_path(path) for path in paths
            ]

            if reference_labels is None:
                reference_labels = labels.copy()
                reference_paths = normalized_paths
                reference_class_names = class_names.copy()

            else:
                if not np.array_equal(labels, reference_labels):
                    raise ValueError(
                        f"Label mismatch detected in {filename}."
                    )

                if normalized_paths != reference_paths:
                    raise ValueError(
                        f"Image path/order mismatch detected in {filename}."
                    )

                if not np.array_equal(
                    class_names,
                    reference_class_names
                ):
                    raise ValueError(
                        f"Class names mismatch detected in {filename}."
                    )

            feature_arrays.append(features.astype(np.float32))

            print(
                f"Loaded {model_name}: {features.shape}"
            )

    # Combine features horizontally.
    fused_features = np.concatenate(
        feature_arrays,
        axis=1
    )

    expected_total_dimension = sum(
        EXPECTED_DIMENSIONS.values()
    )

    if fused_features.shape[1] != expected_total_dimension:
        raise ValueError(
            "Unexpected fused feature dimension."
        )

    output_path = os.path.join(
        output_folder,
        f"{split}.npz"
    )

    np.savez_compressed(
        output_path,
        features=fused_features,
        labels=reference_labels,
        paths=np.asarray(reference_paths),
        class_names=reference_class_names
    )

    print(f"Fused feature shape: {fused_features.shape}")
    print(f"Saved: {output_path}")


def fuse_dataset(dataset_name):

    project_root = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    print("\n===================================")
    print(f"FEATURE FUSION: {dataset_name.upper()}")
    print("===================================")

    for split in SPLITS:
        fuse_split(
            project_root,
            dataset_name,
            split
        )

    print(f"\nFeature fusion completed: {dataset_name}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=DATASETS,
        required=True
    )

    args = parser.parse_args()

    fuse_dataset(args.dataset)