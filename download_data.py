"""Download the Oxford-IIIT Pet dataset (images + annotations, ~800 MB) via torchvision.

Usage:
    python download_data.py

Files end up in data/oxford-iiit-pet/{images,annotations}.
Already have the Kaggle copy? Put it in that folder instead and skip this script.
"""
import argparse
from pathlib import Path

from torchvision.datasets import OxfordIIITPet

import config


def main():
    parser = argparse.ArgumentParser(description="Download Oxford-IIIT Pet.")
    # torchvision creates <root>/oxford-iiit-pet/, which matches config.DATA_DIR
    parser.add_argument("--root", type=Path, default=config.DATA_DIR.parent)
    args = parser.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)
    OxfordIIITPet(root=str(args.root), split="trainval", download=True)
    print("Dataset ready in:", args.root / "oxford-iiit-pet")


if __name__ == "__main__":
    main()
