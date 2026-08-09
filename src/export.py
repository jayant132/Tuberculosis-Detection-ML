"""Export the trained torch model as a joblib artifact for inference.

The artifact bundles the model plus the preprocessing settings it needs,
so prediction code does not have to read config.yaml at all.
"""

import argparse

import joblib
import torch

from data import load_config
from model import build_model


def export_model(cfg, checkpoint_path, output_path):
    model = build_model(cfg)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"])
    model.eval()

    joblib.dump(
        {
            "model": model,
            "image_size": cfg["image_size"],
            "preprocess": cfg["preprocess"],
        },
        output_path,
    )
    print(f"Exported model to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--output", default="checkpoints/model.joblib")
    args = parser.parse_args()
    export_model(load_config(args.config), args.checkpoint, args.output)


if __name__ == "__main__":
    main()
