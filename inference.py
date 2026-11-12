"""Predict the breed of one image, or every image in a folder.

Usage:
    python inference.py path/to/photo.jpg
    python inference.py path/to/folder --top-k 5
"""
import argparse
from pathlib import Path

import torch
from PIL import Image

import config
from data import eval_transform
from model import get_device, load_class_names, load_trained_model


@torch.no_grad()
def predict(model, image_path, class_names, device, top_k=3):
    """Return the top-k (breed, probability) pairs for one image, best first."""
    image = Image.open(image_path).convert("RGB")  # handles JPG, PNG, WEBP
    batch = eval_transform(image).unsqueeze(0).to(device)

    probs = torch.softmax(model(batch), dim=1)[0]
    top_probs, top_ids = probs.topk(min(top_k, len(class_names)))

    return [(class_names[i], p) for i, p in zip(top_ids.tolist(), top_probs.tolist())]


def main():
    parser = argparse.ArgumentParser(description="Predict pet breed from image(s).")
    parser.add_argument("path", type=Path, help="image file or folder of images")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    if not args.path.exists():
        parser.error(f"{args.path} does not exist")

    if args.path.is_dir():
        image_paths = sorted(
            p for p in args.path.iterdir() if p.suffix.lower() in config.VALID_EXTENSIONS
        )
    else:
        image_paths = [args.path]

    if not image_paths:
        parser.error(f"No images ({', '.join(config.VALID_EXTENSIONS)}) found in {args.path}")

    device = get_device()
    class_names = load_class_names()
    model = load_trained_model(len(class_names), device=device)

    for image_path in image_paths:
        print(image_path.name)
        for rank, (breed, prob) in enumerate(predict(model, image_path, class_names, device, args.top_k), 1):
            print(f"  {rank}. {breed:<30} {prob * 100:6.2f}%")


if __name__ == "__main__":
    main()
