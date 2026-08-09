"""Predict TB vs normal for a single chest X-ray with the joblib model."""

import argparse

import joblib
import torch

import preprocessing as pp


def load_predictor(path):
    return joblib.load(path)


def predict(predictor, image_path):
    """Return (label, positive-class probability)."""
    model = predictor["model"]
    size = predictor["image_size"]
    preprocess = predictor["preprocess"]

    x = pp.prepare(
        image_path,
        size,
        use_clahe=preprocess["use_clahe"],
        clip=preprocess["clahe_clip"],
        tile=preprocess["clahe_tile"],
    )
    with torch.no_grad():
        prob = torch.softmax(model(x.unsqueeze(0)), dim=1)[0, 1].item()

    label = "Tuberculosis" if prob >= 0.5 else "Normal"
    return label, prob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="checkpoints/model.joblib")
    parser.add_argument("image", help="path to a chest X-ray")
    args = parser.parse_args()

    label, prob = predict(load_predictor(args.model), args.image)
    print(f"{label}: {prob:.2%}")


if __name__ == "__main__":
    main()
