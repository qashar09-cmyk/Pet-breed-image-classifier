# Pet Breed Classifier

An image classifier for 37 cat and dog breeds from the Oxford-IIIT Pet dataset. The project includes the trained ResNet-18 checkpoint, test-set evaluation, external-photo evaluation, and single-image/folder inference.

## Model and training setup

- **Architecture:** ResNet-18 initialized with ImageNet-pretrained weights, with its final layer replaced by a 37-class classifier. Transfer learning provides useful visual features for a relatively small dataset.
- **Optimization:** Adam, learning rate `1e-4`, batch size `32`, for up to 10 epochs.
- **Preprocessing:** resize to `224 x 224`, ImageNet normalization; random horizontal flips are used for training only.
- **Checkpoint selection:** save the weights with the best validation accuracy.
- **Split:** the official trainval set is stratified into 2,944 training and 736 validation images; the separate 3,669-image official test set is used only by `evaluate.py`.

The existing checkpoint is `outputs/best_resnet18.pth`; its matching label order is in `outputs/class_names.json`. You do not need to train again to run inference or the evaluation scripts.

## Setup

Requires Python and the packages listed in `requirements.txt`.

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset

The Oxford-IIIT Pet dataset is not included in this repository. Download it locally (approximately 800 MB):

```powershell
python download_data.py
```

The scripts expect the data under `data/oxford-iiit-pet/`. The loader accepts either flat `images/` and `annotations/` folders or the nested Kaggle layout (`images/images/`, `annotations/annotations/`). To use a different location, pass `--data-dir` to `train.py` or `evaluate.py`.

The dataset is excluded from Git by `.gitignore`; do not commit the downloaded dataset.

## Run

```powershell
# Optional: retrain the model
python train.py

# Official test-set accuracy, per-breed report, and confusion matrix
python evaluate.py --num-workers 0

# Evaluate the labelled photos in External_pet_images/Cats and /Dogs
python evaluate_external.py

# Predict one image, or all supported images directly in a folder
python inference.py External_pet_images\Cats\Persian_1.png --top-k 3
python inference.py External_pet_images\Dogs --top-k 3
```

`evaluate.py` saves `outputs/test_classification_report.txt` and `outputs/confusion_matrix.png`. `evaluate_external.py` saves `outputs/external_predictions.csv`. External filenames should use `Breed_number.ext`, such as `Persian_1.png`; supported extensions are JPG, JPEG, PNG, and WEBP.

For a reproducible public repository, include only external photos that you own or are licensed to redistribute.

## Results

### Official Oxford-IIIT Pet test set

**Pending.** The test dataset was not present in the local workspace, so the official test evaluation has not been run here. No test accuracy, per-breed precision/recall, or confusion matrix is claimed. After downloading the dataset, run `python evaluate.py --num-workers 0`; this writes the report and confusion-matrix image listed above.

### External photos

- **Images evaluated:** 36
- **Correct:** 35
- **Accuracy:** 97.22%

This is a small sample and should not be treated as a general performance estimate.

**Failure case:** `chihuahua_1.jpg` was labelled as a Chihuahua but predicted as a miniature pinscher. The photo shows a small, short-haired black-and-tan dog with upright ears. Those visual traits overlap between the two breeds, which may have led the model to confuse them; one example is not enough to establish the cause.

## Next improvements

Run and publish the official test-set metrics and confusion matrix. Expand the external test with more independently sourced, rights-cleared photos per breed, then analyze recurring confusions before changing the model or training setup.
