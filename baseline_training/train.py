import argparse
import copy
import json
import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from config import (
    DEVICE,
    MODEL_DIR,
    RESULTS_DIR,
    RANDOM_SEED,
    HEAD_LEARNING_RATE,
    FINETUNE_LEARNING_RATE,
    WEIGHT_DECAY,
    STAGE1_EPOCHS,
    STAGE2_EPOCHS,
    PATIENCE
)

from dataset_loader import create_dataloaders

from models import (
    build_model,
    freeze_backbone,
    unfreeze_selected_layers,
    count_trainable_parameters
)


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_metrics(labels, predictions):

    labels = np.asarray(labels)
    predictions = np.asarray(predictions)

    tp = np.sum((labels == 0) & (predictions == 0))
    tn = np.sum((labels == 1) & (predictions == 1))
    fp = np.sum((labels == 1) & (predictions == 0))
    fn = np.sum((labels == 0) & (predictions == 1))

    accuracy = (tp + tn) / max(len(labels), 1)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    f1 = (
        2 * precision * recall /
        max(precision + recall, 1e-12)
    )

    return {
        "accuracy": float(accuracy),
        "bleached_precision": float(precision),
        "bleached_recall": float(recall),
        "bleached_f1": float(f1)
    }


def run_epoch(model, loader, criterion, optimizer=None):

    training = optimizer is not None

    model.train() if training else model.eval()

    total_loss = 0
    all_labels = []
    all_predictions = []

    for images, labels in loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):

            outputs = model(images)

            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions = torch.argmax(outputs, dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())

    average_loss = total_loss / len(loader.dataset)

    metrics = calculate_metrics(
        all_labels,
        all_predictions
    )

    return average_loss, metrics


def train_stage(
    model,
    train_loader,
    val_loader,
    criterion,
    learning_rate,
    epochs,
    checkpoint_path
):

    optimizer = AdamW(
        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),
        lr=learning_rate,
        weight_decay=WEIGHT_DECAY
    )

    best_f1 = -1
    best_state = None
    best_epoch = 0
    patience_counter = 0

    for epoch in range(epochs):

        train_loss, train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss, val_metrics = run_epoch(
            model,
            val_loader,
            criterion
        )

        val_f1 = val_metrics["bleached_f1"]

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Accuracy: {val_metrics['accuracy']:.4f} | "
            f"Bleached F1: {val_f1:.4f}"
        )

        if val_f1 > best_f1:

            best_f1 = val_f1
            best_epoch = epoch + 1

            best_state = copy.deepcopy(
                model.state_dict()
            )

            torch.save(
                {
                    "model_state_dict": best_state,
                    "best_val_f1": best_f1,
                    "best_epoch": best_epoch
                },
                checkpoint_path
            )

            patience_counter = 0

        else:

            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("Early stopping.")
            break

    model.load_state_dict(best_state)

    return model, best_f1, best_epoch


def train_experiment(dataset_name, model_name):

    set_seed(RANDOM_SEED)

    train_loader, val_loader, _ = create_dataloaders(
        dataset_name
    )

    model_output_dir = (
        MODEL_DIR / dataset_name / model_name
    )

    result_output_dir = (
        RESULTS_DIR / dataset_name / model_name
    )

    model_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    result_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    model = build_model(model_name).to(DEVICE)

    # Calculate class weights from training data only.
    targets = np.array(train_loader.dataset.targets)

    class_counts = np.bincount(
        targets,
        minlength=2
    )

    if np.any(class_counts == 0):
        raise ValueError("A class is missing from training data.")

    class_weights = len(targets) / (
        2.0 * class_counts
    )

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            class_weights,
            dtype=torch.float32,
            device=DEVICE
        )
    )

    # -------------------------
    # Stage 1: Train classifier
    # -------------------------

    print("\n========== STAGE 1 ==========")

    freeze_backbone(model, model_name)

    print(
        "Trainable parameters:",
        count_trainable_parameters(model)
    )

    stage1_path = model_output_dir / "stage1_best.pth"

    model, stage1_f1, stage1_epoch = train_stage(
        model,
        train_loader,
        val_loader,
        criterion,
        HEAD_LEARNING_RATE,
        STAGE1_EPOCHS,
        stage1_path
    )

    # ---------------------------------
    # Stage 2: Fine-tune selected layers
    # ---------------------------------

    print("\n========== STAGE 2 ==========")

    unfreeze_selected_layers(model, model_name)

    print(
        "Trainable parameters:",
        count_trainable_parameters(model)
    )

    stage2_path = model_output_dir / "stage2_best.pth"

    model, stage2_f1, stage2_epoch = train_stage(
        model,
        train_loader,
        val_loader,
        criterion,
        FINETUNE_LEARNING_RATE,
        STAGE2_EPOCHS,
        stage2_path
    )

    # ---------------------------------
    # Select best stage using validation
    # ---------------------------------

    if stage2_f1 > stage1_f1:

        selected_path = stage2_path
        selected_stage = "stage2_finetuned"
        selected_f1 = stage2_f1

    else:

        selected_path = stage1_path
        selected_stage = "stage1_frozen"
        selected_f1 = stage1_f1

    selected_checkpoint = torch.load(
        selected_path,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(
        selected_checkpoint["model_state_dict"]
    )

    final_path = model_output_dir / "best_model.pth"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": model_name,
            "dataset_name": dataset_name,
            "selected_stage": selected_stage,
            "best_val_f1": selected_f1,
            "class_to_idx": {
                "bleached": 0,
                "healthy": 1
            }
        },
        final_path
    )

    summary = {
        "dataset": dataset_name,
        "model": model_name,
        "stage1_best_f1": stage1_f1,
        "stage1_best_epoch": stage1_epoch,
        "stage2_best_f1": stage2_f1,
        "stage2_best_epoch": stage2_epoch,
        "selected_stage": selected_stage,
        "selected_validation_f1": selected_f1
    }

    with open(
        result_output_dir / "training_summary.json",
        "w"
    ) as file:

        json.dump(summary, file, indent=4)

    print("\nTraining completed.")
    print(json.dumps(summary, indent=4))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "kaggle_only",
            "self_collected_only",
            "combined"
        ]
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "resnet50",
            "efficientnet_b0",
            "vit_small"
        ]
    )

    args = parser.parse_args()

    train_experiment(
        args.dataset,
        args.model
    )