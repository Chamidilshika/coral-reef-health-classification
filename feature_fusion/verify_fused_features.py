import os
import argparse
import numpy as np

EXPECTED_DIMENSION = 3712

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
        "fused_features",
        dataset_name
    )

    print(f"\n========== {dataset_name.upper()} ==========")

    all_passed = True

    for split, expected_count in EXPECTED_SAMPLES[dataset_name].items():

        filename = f"{split}.npz"
        file_path = os.path.join(feature_folder, filename)

        if not os.path.exists(file_path):
            print(f"FAILED: Missing {filename}")
            all_passed = False
            continue

        try:
            with np.load(file_path) as data:

                features = data["features"]
                labels = data["labels"]
                paths = data["paths"]
                class_names = data["class_names"]

                valid = (
                    features.shape == (
                        expected_count,
                        EXPECTED_DIMENSION
                    )
                    and len(labels) == expected_count
                    and len(paths) == expected_count
                    and np.isfinite(features).all()
                    and len(class_names) == 2
                    and len(np.unique(labels)) == 2
                    and len(set(paths.tolist())) == expected_count
                )

                if valid:
                    print(
                        f"PASSED: {filename} "
                        f"| Shape: {features.shape}"
                    )
                else:
                    print(
                        f"FAILED: {filename} "
                        f"| Shape: {features.shape}"
                    )
                    all_passed = False

        except Exception as error:
            print(f"FAILED: {filename} | Error: {error}")
            all_passed = False

    if all_passed:
        print("\nAll fused feature files passed verification!")
    else:
        print("\nSome fused feature files need attention.")

    return all_passed


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=list(EXPECTED_SAMPLES.keys()),
        required=True
    )

    args = parser.parse_args()

    verify_dataset(args.dataset)
