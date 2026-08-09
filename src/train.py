"""Training loop: class-weighted loss, early stopping, best checkpoint."""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score

from data_processing import get_dataloaders, load_config
from model import build_model


def pick_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def evaluate(model, loader, criterion, device):
    """Return (true labels, positive-class probabilities, mean loss)."""
    model.eval()
    losses, ys, probs = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            losses.append(criterion(logits, y).item() * len(x))
            probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            ys.append(y.cpu().numpy())
    ys = np.concatenate(ys)
    probs = np.concatenate(probs)
    return ys, probs, np.sum(losses) / len(ys)


def train(cfg, device=None):
    device = device or pick_device()
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    data = get_dataloaders(cfg)
    model = build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss(weight=data.class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("experiments", exist_ok=True)
    tcfg = cfg["train"]

    best_loss = float("inf")
    wait = 0
    rows = []
    for epoch in range(tcfg["epochs"]):
        model.train()
        train_loss = 0.0
        for x, y in data.train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x)
        train_loss /= len(data.train_df)

        _, val_probs, val_loss = evaluate(model, data.val_loader, criterion, device)
        val_pr_auc = average_precision_score(data.val_df["label_int"], val_probs)

        rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_pr_auc": val_pr_auc,
            }
        )
        pd.DataFrame(rows).to_csv("experiments/run.csv", index=False)
        print(
            f"epoch {epoch:3d}  train_loss {train_loss:.4f}  "
            f"val_loss {val_loss:.4f}  val_pr_auc {val_pr_auc:.4f}"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            wait = 0
            torch.save(
                {"state_dict": model.state_dict(), "epoch": epoch, "val_loss": val_loss},
                "checkpoints/best.pt",
            )
        else:
            wait += 1
            if wait >= tcfg["early_stop_patience"]:
                print(f"Early stopping at epoch {epoch}")
                break

    return model, data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
