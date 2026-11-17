
import argparse
import json

import numpy as np
import torch
from pathlib import Path

from config import (
    DEVICE,
    MODEL_DIR,
    RESULTS_DIR
)

from dataset_loader import create_dataloaders
from models import build_model

from train import calculate_metrics


def evaluate_model(dataset_name, model_name):

    _, _, test_loader = create_dataloaders(
        dataset_name
    )

    checkpoint_path = (
        MODEL_DIR /
        dataset_name /
        model_name /
        "best_model.pth"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True
    )

    model = build_model(model_name).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)

            outputs = model(images)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            predictions = torch.argmax(
                probabilities,
                dim=1
            )

            all_labels.extend(labels.numpy())
            all_predictions.extend(
                predictions.cpu().numpy()
            )

            # Probability of bleached class (label 0)
            all_probabilities.extend(
                probabilities[:, 0].cpu().numpy()
            )

    metrics = calculate_metrics(
        all_labels,
        all_predictions
    )

    labels = np.asarray(all_labels)
    predictions = np.asarray(all_predictions)

    tp = int(np.sum((labels == 0) & (predictions == 0)))
    tn = int(np.sum((labels == 1) & (predictions == 1)))
    fp = int(np.sum((labels == 1) & (predictions == 0)))
    fn = int(np.sum((labels == 0) & (predictions == 1)))

    metrics["confusion_matrix"] = [
        [tp, fn],
        [fp, tn]
    ]

    metrics["bleached_roc_auc"] = None

    try:

        from sklearn.metrics import roc_auc_score

        metrics["bleached_roc_auc"] = float(
            roc_auc_score(
                (labels == 0).astype(int),
                all_probabilities
            )
        )

    except Exception as error:

        print(
            "ROC-AUC could not be calculated:",
            error
        )

    output_dir = (
        RESULTS_DIR / dataset_name / model_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = output_dir / "test_results.json"

    with open(output_file, "w") as file:

        json.dump(metrics, file, indent=4)

    print("\nTEST RESULTS")
    print(json.dumps(metrics, indent=4))

    print("\nSaved:", output_file)


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

    evaluate_model(
        args.dataset,
        args.model
    )