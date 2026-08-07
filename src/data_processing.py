import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_metadata(root):
    df = pd.read_csv(os.path.join(root, "metadata.csv"))
    df["label_int"] = df["label"].map(
        {label: i for i, label in enumerate(sorted(df["label"].unique()))}
    )
    return df


def validate_files(df, root):
    missing, corrupt = [], []
    for path in df["filename"]:
        full = os.path.join(root, path)
        if not os.path.exists(full):
            missing.append(path)
            continue
        try:
            with Image.open(full) as im:
                im.load()
        except Exception:
            corrupt.append(path)
    bad = set(missing) | set(corrupt)
    clean = df[~df["filename"].isin(bad)].reset_index(drop=True)
    if missing:
        print(f"Missing files: {len(missing)}")
    if corrupt:
        print(f"Corrupt files: {len(corrupt)}")
    return clean, missing, corrupt


def stratified_split(df, val_ratio, test_ratio, seed):
    train, test = train_test_split(
        df, test_size=test_ratio, stratify=df["label_int"], random_state=seed
    )
    val_frac = val_ratio / (1 - test_ratio)
    train, val = train_test_split(
        train, test_size=val_frac, stratify=train["label_int"], random_state=seed
    )
    return (split.reset_index(drop=True) for split in (train, val, test))


def compute_class_weights(df):
    counts = df["label_int"].value_counts().sort_index().to_numpy(dtype=np.float32)
    total = counts.sum()
    return torch.tensor(total / (len(counts) * counts), dtype=torch.float32)


class XRayDataset(Dataset):
    def __init__(self, df, root, size):
        self.df = df
        self.root = root
        self.size = size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = self._load(os.path.join(self.root, row["filename"]))
        return x, int(row["label_int"])

    def _load(self, path):
        with Image.open(path) as im:
            im = im.convert("L").resize((self.size, self.size), Image.BILINEAR)
            arr = np.array(im, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    class_weights: torch.Tensor


def get_dataloaders(cfg):
    root = cfg["data"]["root"]
    df = load_metadata(root)
    df, _, _ = validate_files(df, root)

    split = cfg["split"]
    train_df, val_df, test_df = stratified_split(
        df, split["val_ratio"], split["test_ratio"], cfg["seed"]
    )

    dataset_args = {"root": root, "size": cfg["image_size"]}
    train_loader = DataLoader(
        XRayDataset(train_df, **dataset_args),
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
    )
    eval_loader = DataLoader(
        XRayDataset(val_df, **dataset_args),
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
    )
    test_loader = DataLoader(
        XRayDataset(test_df, **dataset_args),
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
    )

    return DataBundle(
        train_loader, eval_loader, test_loader,
        train_df, val_df, test_df,
        compute_class_weights(train_df),
    )
