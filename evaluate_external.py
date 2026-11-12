"""Generalisation check on your own photos.

Expected layout (breed is read from the filename, e.g. Persian_1.png -> Persian):

    data/External_pet_images/
        Cats/  Persian_1.png, Bengal_2.jpg, ...
        Dogs/  Beagle_1.jpg, Pug_3.webp, ...

Usage:
    python evaluate_external.py
    python evaluate_external.py --external-dir path/to/External_pet_images
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

import config
from inference import predict
from model import get_device, load_class_names, load_trained_model


def main():
    parser = argparse.ArgumentParser(description="Test the model on your own labelled photos.")
    parser.add_argument("--external-dir", type=Path, default=config.EXTERNAL_DIR)
    args = parser.parse_args()

    device = get_device()
    class_names = load_class_names()
    model = load_trained_model(len(class_names), device=device)

    # Case-insensitive lookup: the dataset has "Persian" (cats) but "beagle" (dogs, lowercase)
    name_lookup = {name.lower(): name for name in class_names}

    results = []
    for folder in ("Cats", "Dogs"):
        folder_path = args.external_dir / folder
        if not folder_path.is_dir():
            print(f"Folder not found, skipping: {folder_path}")
            continue

        for image_path in sorted(folder_path.iterdir()):
            if image_path.suffix.lower() not in config.VALID_EXTENSIONS:
                continue

            # Persian_1.png -> Persian
            actual_key = image_path.stem.rsplit("_", 1)[0].lower()
            if actual_key not in name_lookup:
                print("Breed not found in training classes:", image_path.name)
                continue

            actual_breed = name_lookup[actual_key]
            predicted_breed = predict(model, image_path, class_names, device, top_k=1)[0][0]

            results.append({
                "filename": image_path.name,
                "actual_breed": actual_breed,
                "predicted_breed": predicted_breed,
                "correct": actual_breed == predicted_breed,
            })

    if not results:
        raise SystemExit(f"No usable images found in {args.external_dir}")

    results_df = pd.DataFrame(results)
    accuracy = accuracy_score(results_df["actual_breed"], results_df["predicted_breed"])

    print("\nExternal Image Testing Results")
    print("--------------------------------")
    print("Total images tested:", len(results_df))
    print(f"External Accuracy: {accuracy * 100:.2f}%")

    # Only breeds that actually appear in your photos
    present_breeds = sorted(results_df["actual_breed"].unique())
    print("\nClassification Report (Present Breeds Only):\n")
    print(classification_report(
        results_df["actual_breed"],
        results_df["predicted_breed"],
        labels=present_breeds,
        zero_division=0,
    ))

    print("\nIndividual Predictions:\n")
    print(results_df.to_string(index=False))

    print("\nIncorrect Predictions:")
    print(results_df[~results_df["correct"]].to_string(index=False))

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(config.EXTERNAL_PREDICTIONS_PATH, index=False)
    print(f"\nResults saved: {config.EXTERNAL_PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
