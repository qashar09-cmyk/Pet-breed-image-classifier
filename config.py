"""Central settings: paths and hyperparameters. Edit values here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Dataset locations (see README)
DATA_DIR = ROOT / "data" / "oxford-iiit-pet"  # contains images/ and annotations/
EXTERNAL_DIR = ROOT / "External_pet_images"    # contains Cats/ and Dogs/

# Outputs
OUTPUT_DIR = ROOT / "outputs"
BEST_MODEL_PATH = OUTPUT_DIR / "best_resnet18.pth"
CLASS_NAMES_PATH = OUTPUT_DIR / "class_names.json"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "confusion_matrix.png"
TEST_REPORT_PATH = OUTPUT_DIR / "test_classification_report.txt"
EXTERNAL_PREDICTIONS_PATH = OUTPUT_DIR / "external_predictions.csv"
EXTERNAL_RESULTS_IMAGE_PATH = OUTPUT_DIR / "external_evaluation_results.png"

# Data
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 2        # set to 0 if DataLoader gives trouble on your machine
VAL_SPLIT = 0.20
SEED = 42
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

# Training
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
