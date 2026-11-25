from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASETS = [
    "kaggle_only",
    "self_collected_only",
    "combined"
]

MODELS = [
    "resnet50",
    "efficientnet_b0",
    "vit_small"
]

SPLITS = ["train", "val", "test"]

EXPECTED_FEATURE_DIMENSIONS = {
    "resnet50": 2048,
    "efficientnet_b0": 1280,
    "vit_small": 384
}

EXPECTED_SPLIT_COUNTS = {
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

EXPERIMENTS_DIR = PROJECT_ROOT / "dataset" / "experiments"

CHECKPOINT_DIR = PROJECT_ROOT / "trained_models"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "extracted_features"