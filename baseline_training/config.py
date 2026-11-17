from pathlib import Path
import torch

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset directory
EXPERIMENTS_DIR = PROJECT_ROOT / "dataset" / "experiments"

# Output directories
MODEL_DIR = PROJECT_ROOT / "trained_models"
RESULTS_DIR = PROJECT_ROOT / "baseline_results"

# Available datasets and models
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

# Class mapping
CLASS_TO_IDX = {
    "bleached": 0,
    "healthy": 1
}

# Reproducibility
RANDOM_SEED = 42

# Training settings
IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 0  # Safe setting for Windows
STAGE1_EPOCHS = 15
STAGE2_EPOCHS = 15

# Learning rates
HEAD_LEARNING_RATE = 0.001
FINETUNE_LEARNING_RATE = 0.0001

WEIGHT_DECAY = 0.0001
PATIENCE = 5

# Device
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {DEVICE}")