"""Model helpers: build ResNet-18, load trained weights, save/load class names."""
import json

import torch
import torch.nn as nn
from torchvision import models

import config


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(num_classes, pretrained=True):
    """ResNet-18 with the final layer replaced for `num_classes` outputs."""
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_trained_model(num_classes, path=config.BEST_MODEL_PATH, device=None):
    """Rebuild the architecture and load the saved best weights, ready for inference."""
    device = device or get_device()
    if not path.exists():
        raise FileNotFoundError(f"No trained weights at {path}. Run `python train.py` first.")

    model = build_model(num_classes, pretrained=False)  # no need to download ImageNet weights
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.to(device).eval()


def save_class_names(class_names, path=config.CLASS_NAMES_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(class_names, indent=2))


def load_class_names(path=config.CLASS_NAMES_PATH):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python train.py` first.")
    return json.loads(path.read_text())
