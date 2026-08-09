"""Streamlit UI for TB detection from chest X-rays.

Run with:  streamlit run app.py
"""

import os
import sys

import cv2
import joblib
import numpy as np
import streamlit as st
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import preprocessing as pp

MODEL_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "model.joblib")
LABELS = {0: "Normal", 1: "Tuberculosis"}


@st.cache_resource
def load_predictor(path):
    return joblib.load(path)


def decode_upload(uploaded):
    data = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def main():
    st.set_page_config(page_title="TB Detection", page_icon="🫁", layout="centered")
    st.title("Tuberculosis Detection from Chest X-rays")
    st.write(
        "Upload a chest X-ray and the tiny CNN will tell you if it looks "
        "normal or TB-positive."
    )

    with st.spinner("Loading model..."):
        predictor = load_predictor(MODEL_PATH)

    uploaded = st.file_uploader("Choose a chest X-ray", type=["png", "jpg", "jpeg"])
    if uploaded is None:
        st.info("Upload an X-ray to get a prediction.")
        return

    image = decode_upload(uploaded)
    if image is None:
        st.error("Could not read that file as an image.")
        return

    model = predictor["model"]
    size = predictor["image_size"]
    preprocess = predictor["preprocess"]

    col1, col2 = st.columns(2)
    col1.image(image, caption="Raw upload", use_container_width=True)

    tensor = pp.prepare_image(
        image,
        size,
        use_clahe=preprocess["use_clahe"],
        clip=preprocess["clahe_clip"],
        tile=preprocess["clahe_tile"],
    )
    if preprocess["use_clahe"]:
        col2.image(tensor.squeeze().numpy(), caption="After CLAHE", use_container_width=True)

    with torch.no_grad():
        prob = torch.softmax(model(tensor.unsqueeze(0)), dim=1)[0, 1].item()

    label = LABELS[int(prob >= 0.5)]
    st.markdown(f"## Prediction: **{label}**")
    st.progress(float(prob))
    st.caption(f"TB probability: {prob:.1%} · Normal probability: {1 - prob:.1%}")


if __name__ == "__main__":
    main()
