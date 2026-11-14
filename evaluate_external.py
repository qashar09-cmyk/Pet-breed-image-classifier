"""Generalisation check on your own photos.

Expected layout (breed is read from the filename, e.g. Persian_1.png -> Persian):

    data/External_pet_images/
        Cats/  Persian_1.png, Bengal_2.jpg, ...
        Dogs/  Beagle_1.jpg, Pug_3.webp, ...

Usage:
    python evaluate_external.py
    python evaluate_external.py --external-dir path/to/External_pet_images

The predictions are saved as CSV, and the complete printed report is saved as an image.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

import config
from inference import predict
from model import get_device, load_class_names, load_trained_model


def save_results_image(output_text, path):
    lines = output_text.splitlines()
    figure_height = max(6, len(lines) * 0.22)
    figure, axis = plt.subplots(figsize=(12, figure_height), facecolor="#1e1e1e")
    axis.set_facecolor("#1e1e1e")
    axis.axis("off")

    line_step = 0.96 / max(1, len(lines))
    for index, line in enumerate(lines):
        axis.text(
            0.02,
            0.98 - index * line_step,
            line,
            color="#d4d4d4",
            fontfamily="monospace",
            fontsize=9,
            va="top",
            transform=axis.transAxes,
        )

    figure.savefig(path, dpi=150, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


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

    # Only breeds that actually appear in your photos
    present_breeds = sorted(results_df["actual_breed"].unique())
    report = classification_report(
        results_df["actual_breed"],
        results_df["predicted_breed"],
        labels=present_breeds,
        zero_division=0,
    ).rstrip()

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(config.EXTERNAL_PREDICTIONS_PATH, index=False)

    incorrect_predictions = results_df[~results_df["correct"]]
    output_lines = [
        "",
        "External Image Testing Results",
        "--------------------------------",
        f"Total images tested: {len(results_df)}",
        f"External Accuracy: {accuracy * 100:.2f}%",
        "",
        "Classification Report (Present Breeds Only):",
        "",
        *report.splitlines(),
        "",
        "Individual Predictions:",
        "",
        *results_df.to_string(index=False).splitlines(),
        "",
        "Incorrect Predictions:",
    ]
    if incorrect_predictions.empty:
        output_lines.append("None")
    else:
        output_lines.extend(incorrect_predictions.to_string(index=False).splitlines())
    output_lines.extend([
        "",
        f"Results saved: {config.EXTERNAL_PREDICTIONS_PATH}",
        f"Results image saved: {config.EXTERNAL_RESULTS_IMAGE_PATH}",
    ])
    output_text = "\n".join(output_lines) + "\n"
    save_results_image(output_text, config.EXTERNAL_RESULTS_IMAGE_PATH)
    print(output_text, end="")


if __name__ == "__main__":
    main()
