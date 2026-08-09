"""Evaluation of a trained checkpoint across all splits."""

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from data_processing import get_dataloaders, load_config
from model import build_model
from train import evaluate, pick_device


def split_metrics(y_true, y_prob, labels=(0, 1)):
    y_pred = (y_prob >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "confusion": confusion_matrix(y_true, y_pred, labels=labels),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_checkpoint(cfg, checkpoint_path, device=None):
    device = device or pick_device()
    data = get_dataloaders(cfg)
    model = build_model(cfg).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["state_dict"])

    criterion = nn.CrossEntropyLoss(weight=data.class_weights.to(device))
    results = {}
    for name in ("train", "val", "test"):
        loader = getattr(data, f"{name}_loader")
        y_true, y_prob, _ = evaluate(model, loader, criterion, device)
        results[name] = split_metrics(y_true, y_prob)
    return results


def format_report(results, labels=("Normal", "Tuberculosis")):
    splits = list(results)
    data = {}
    for i, label in enumerate(labels):
        for metric in ("precision", "recall", "f1"):
            data[f"{metric} ({label})"] = [results[s][metric][i] for s in splits]
    data["accuracy"] = [results[s]["accuracy"] for s in splits]
    data["roc_auc"] = [results[s]["roc_auc"] for s in splits]
    data["pr_auc"] = [results[s]["pr_auc"] for s in splits]
    return pd.DataFrame(data, index=splits).T.round(4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    args = parser.parse_args()

    results = evaluate_checkpoint(load_config(args.config), args.checkpoint)
    print(format_report(results))
    print()
    for split, m in results.items():
        print(f"{split} confusion matrix (rows=true, cols=pred):")
        print(m["confusion"])
        print()


if __name__ == "__main__":
    main()
