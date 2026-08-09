"""Dataset loading, cleaning and splitting.

Images are decoded and preprocessed by ``preprocessing.py`` (OpenCV);
this module only worries about the metadata, the split and the loaders.
"""

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

import preprocessing as pp


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
        if not os.path.isfile(full):
            missing.append(path)
        elif not pp.is_readable(full):
            corrupt.append(path)
    bad = set(missing) | set(corrupt)
    clean = df[~df["filename"].isin(bad)].reset_index(drop=True)
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
    def __init__(self, df, root, size, preprocess):
        self.df = df
        self.root = root
        self.size = size
        self.preprocess = preprocess

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.root, row["filename"])
        x = pp.prepare(
            path,
            self.size,
            use_clahe=self.preprocess["use_clahe"],
            clip=self.preprocess["clahe_clip"],
            tile=self.preprocess["clahe_tile"],
        )
        return x, int(row["label_int"])


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

    dataset_args = {
        "root": root,
        "size": cfg["image_size"],
        "preprocess": cfg["preprocess"],
    }
    batch = cfg["train"]["batch_size"]
    workers = cfg["num_workers"]

    train_loader = DataLoader(
        XRayDataset(train_df, **dataset_args),
        batch_size=batch,
        shuffle=True,
        num_workers=workers,
    )
    val_loader = DataLoader(
        XRayDataset(val_df, **dataset_args),
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
    )
    test_loader = DataLoader(
        XRayDataset(test_df, **dataset_args),
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
    )

    return DataBundle(
        train_loader, val_loader, test_loader,
        train_df, val_df, test_df,
        compute_class_weights(train_df),
    )
