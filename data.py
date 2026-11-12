"""Data handling: read annotations, split train/val/test, transforms, DataLoaders."""
import os

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import config

_normalize = transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)

train_transform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    _normalize,
])

# Used for validation, test, external images and inference (no augmentation).
eval_transform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    _normalize,
])


def _find_dir(base, name):
    """The Kaggle copy nests folders (images/images); the official download doesn't."""
    for path in (os.path.join(base, name, name), os.path.join(base, name)):
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(
        f"Could not find a '{name}' folder inside {base}. "
        "Run `python download_data.py` or change DATA_DIR in config.py."
    )


def load_annotations(filename, data_dir=config.DATA_DIR):
    """Read trainval.txt / test.txt into a DataFrame with columns: image, breed."""
    image_dir = _find_dir(data_dir, "images")
    annotation_dir = _find_dir(data_dir, "annotations")

    records = []
    with open(os.path.join(annotation_dir, filename), "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            image_id = line.split()[0]                # e.g. Persian_12
            records.append({
                "image": os.path.join(image_dir, image_id + ".jpg"),
                "breed": image_id.rsplit("_", 1)[0],  # e.g. Persian
            })

    return pd.DataFrame(records)


def prepare_splits(data_dir=config.DATA_DIR, val_split=config.VAL_SPLIT, seed=config.SEED):
    """Official trainval -> train + val (stratified by breed). Official test stays untouched."""
    trainval_df = load_annotations("trainval.txt", data_dir)
    test_df = load_annotations("test.txt", data_dir)

    train_df, val_df = train_test_split(
        trainval_df,
        test_size=val_split,
        random_state=seed,
        stratify=trainval_df["breed"],
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    for name, df in (("train", train_df), ("val", val_df), ("test", test_df)):
        missing = [p for p in df["image"] if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"{len(missing)} {name} images are missing, e.g. {missing[0]}")

    class_names = sorted(trainval_df["breed"].unique())

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)} | Classes: {len(class_names)}")
    return train_df, val_df, test_df, class_names


class PetDataset(Dataset):
    def __init__(self, dataframe, class_to_idx, transform=None):
        self.dataframe = dataframe
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        image = Image.open(row["image"]).convert("RGB")
        label = self.class_to_idx[row["breed"]]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_dataloaders(data_dir=config.DATA_DIR, batch_size=config.BATCH_SIZE,
                    num_workers=config.NUM_WORKERS):
    """Returns train_loader, val_loader, test_loader, class_names."""
    train_df, val_df, test_df, class_names = prepare_splits(data_dir)
    class_to_idx = {breed: i for i, breed in enumerate(class_names)}

    train_loader = DataLoader(
        PetDataset(train_df, class_to_idx, train_transform),
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
    )
    val_loader = DataLoader(
        PetDataset(val_df, class_to_idx, eval_transform),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    test_loader = DataLoader(
        PetDataset(test_df, class_to_idx, eval_transform),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader, class_names
