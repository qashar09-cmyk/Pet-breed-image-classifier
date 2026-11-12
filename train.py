"""Fine-tune ResNet-18 on Oxford-IIIT Pet. Saves the best checkpoint by validation accuracy.

Usage:
    python train.py
    python train.py --epochs 15 --lr 0.0001 --batch-size 64
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import config
from data import get_dataloaders
from model import build_model, get_device, save_class_names


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None):
    """One pass over `loader`. Trains if an optimizer is given, otherwise just evaluates.

    Returns (average loss per batch, accuracy in %).
    """
    training = optimizer is not None
    model.train(training)

    running_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if training:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return running_loss / len(loader), 100 * correct / total


def main():
    parser = argparse.ArgumentParser(description="Train the pet breed classifier.")
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--no-pretrained", action="store_true",
                        help="train from random weights instead of ImageNet weights")
    args = parser.parse_args()

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(config.SEED)

    device = get_device()
    print("Using device:", device)

    # The test loader is deliberately ignored here: test data is never used for training decisions.
    train_loader, val_loader, _, class_names = get_dataloaders(
        args.data_dir, args.batch_size, args.num_workers
    )
    save_class_names(class_names)

    model = build_model(len(class_names), pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_accuracy = 0.0

    for epoch in range(args.epochs):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)

        print(f"Epoch [{epoch + 1}/{args.epochs}]")
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Training Accuracy: {train_acc:.2f}%")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_acc:.2f}%")
        print("-" * 40)

        # Keep the checkpoint with the best validation accuracy
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)
            print("Best model saved!")

    print("Training completed!")
    print(f"Best Validation Accuracy: {best_val_accuracy:.2f}%")
    print(f"Weights saved to: {config.BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
