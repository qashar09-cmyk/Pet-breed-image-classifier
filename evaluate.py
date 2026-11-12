"""Evaluate the best saved model on the official Oxford-IIIT Pet test set.

Prints test accuracy and per-breed precision/recall/F1, and saves the confusion matrix.

Usage:
    python evaluate.py
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save to file, no GUI window needed
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import config
from data import get_dataloaders
from model import get_device, load_trained_model


def save_confusion_matrix(y_true, y_pred, class_names, path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    plt.figure(figsize=(18, 16))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix - ResNet-18")
    plt.colorbar()
    plt.xticks(np.arange(len(class_names)), class_names, rotation=90, fontsize=7)
    plt.yticks(np.arange(len(class_names)), class_names, fontsize=7)
    plt.xlabel("Predicted Breed")
    plt.ylabel("Actual Breed")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate on the official test set.")
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    args = parser.parse_args()

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print("Using device:", device)

    _, _, test_loader, class_names = get_dataloaders(args.data_dir, num_workers=args.num_workers)
    model = load_trained_model(len(class_names), device=device)

    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(device))
            y_pred.extend(outputs.argmax(dim=1).cpu().numpy())
            y_true.extend(labels.numpy())

    test_accuracy = accuracy_score(y_true, y_pred)
    print(f"\nOverall Test Accuracy: {test_accuracy * 100:.2f}%")

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
        digits=4,
    )
    print("\nClassification Report:\n")
    print(report)

    config.TEST_REPORT_PATH.write_text(
        f"Overall Test Accuracy: {test_accuracy * 100:.2f}%\n\n{report}"
    )
    save_confusion_matrix(y_true, y_pred, class_names, config.CONFUSION_MATRIX_PATH)

    print(f"Saved: {config.TEST_REPORT_PATH}")
    print(f"Saved: {config.CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    main()
