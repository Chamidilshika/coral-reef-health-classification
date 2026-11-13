
import os
import argparse
import numpy as np


# Expected feature dimensions
EXPECTED_DIMENSIONS = {
    "resnet50": 2048,
    "efficientnet_b0": 1280,
    "vit_small": 384
}


# Expected image counts for each dataset
EXPECTED_SAMPLES = {
    "kaggle_only": {
        "train": 634,
        "val": 136,
        "test": 137
    },

    "self_collected_only": {
        "train": 225,
        "val": 49,
        "test": 49
    },

    "combined": {
        "train": 859,
        "val": 185,
        "test": 186
    }
}


def verify_dataset(dataset_name):

    project_root = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    feature_folder = os.path.join(
        project_root,
        "extracted_features",
        dataset_name
    )

    print(f"\n========== {dataset_name.upper()} ==========")

    all_passed = True

    # Get expected counts for the selected dataset
    expected_counts = EXPECTED_SAMPLES[dataset_name]

    for split, expected_count in expected_counts.items():

        for model_name, expected_dimension in EXPECTED_DIMENSIONS.items():

            filename = f"{model_name}_{split}.npz"

            file_path = os.path.join(
                feature_folder,
                filename
            )

            if not os.path.exists(file_path):

                print(f"FAILED: Missing {filename}")
                all_passed = False
                continue

            with np.load(file_path) as data:

                features = data["features"]
                labels = data["labels"]
                paths = data["paths"]

                expected_shape = (
                    expected_count,
                    expected_dimension
                )

                valid = (
                    features.shape == expected_shape
                    and len(labels) == expected_count
                    and len(paths) == expected_count
                    and np.isfinite(features).all()
                )

                if valid:

                    print(
                        f"PASSED: {filename} "
                        f"| Shape: {features.shape}"
                    )

                else:

                    print(f"FAILED: {filename}")

                    print(
                        f"Expected shape: {expected_shape}"
                    )

                    print(
                        f"Actual shape: {features.shape}"
                    )

                    print(
                        f"Labels: {len(labels)}"
                    )

                    print(
                        f"Paths: {len(paths)}"
                    )

                    all_passed = False

    if all_passed:

        print("\nAll feature files passed verification!")

    else:

        print("\nSome feature files need attention.")

    return all_passed


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=[
            "kaggle_only",
            "self_collected_only",
            "combined"
        ]
    )

    args = parser.parse_args()

    verify_dataset(args.dataset)
