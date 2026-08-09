# Tuberculosis Detection from Chest X-Rays

A lightweight, deployment-ready deep learning project that classifies chest X-rays as **Normal** or **Tuberculosis**. Images are preprocessed with **OpenCV** (CLAHE contrast enhancement), classified by a **tiny 60K-parameter CNN** that trains on a laptop CPU in under a minute, and served through a **Streamlit** web app that runs locally or in **Docker**.

![Python](https://img.shields.io/badge/Python-3.12-3776AB) ![PyTorch](https://img.shields.io/badge/PyTorch-2.13-EE4C2C) ![OpenCV](https://img.shields.io/badge/OpenCV-5.0-5C3EE8) ![Streamlit](https://img.shields.io/badge/Streamlit-1.61-FF4B4B) ![License](https://img.shields.io/badge/License-MIT-green)

---

## Why this project

Most X-ray classifiers reach for massive pretrained networks (50M+ parameters) that take a GPU to train and hours to run. This project deliberately goes the other way: a **~60K-parameter CNN** that hits **83% test accuracy** and **0.91 test PR-AUC** on a 400-image dataset while training in seconds on a CPU. It is a clean, honest example of:

- **End-to-end ML engineering** - data loading, OpenCV preprocessing, training, evaluation, export, inference and a web UI.
- **Production thinking** - config-driven, deterministic seeds, early stopping, reproducible artifact export (joblib), Docker packaging.
- **Small data, small model** - how to build a practical medical-imaging classifier when you don't have a GPU or a million images.

## Key features

| Feature | What it does |
| --- | --- |
| **OpenCV preprocessing** | Grayscale load, CLAHE local-contrast boost, resizing, normalization |
| **Tiny CNN** | 3 conv blocks + global average pooling, ~60K parameters |
| **Data augmentation** | Random flips, rotations and brightness jitter (OpenCV-based) for training |
| **Class-aware training** | Weighted cross-entropy + early stopping + best-checkpoint saving |
| **Full metrics** | Accuracy, ROC-AUC, PR-AUC, precision/recall/F1, confusion matrices |
| **Deployable** | Streamlit UI, joblib model artifact, Docker + Docker Compose |

## Tech stack

- **Python 3.12** · **PyTorch** (CPU) · **OpenCV** · **NumPy/Pandas** · **scikit-learn** · **Streamlit** · **Joblib** · **Docker**

## Architecture

```
                         ┌──────────────────────────────────────────┐
  Chest X-ray ──────────▶│  OpenCV pipeline (src/preprocessing.py)  │
   (256×256, gray)       │  load → CLAHE → resize(128) → normalize   │
                         └──────────────────┬───────────────────────┘
                                            ▼
                         ┌──────────────────────────────────────────┐
                         │  TBCNN  (src/model.py) ~60K params       │
                         │  Conv(32) → Conv(64) → Conv(64) → GAP     │
                         └──────────────────┬───────────────────────┘
                                            ▼
                         ┌──────────────────────────────────────────┐
                         │   Normal │ Tuberculosis + probability    │
                         └──────────────────────────────────────────┘
```

Training writes the best checkpoint to `checkpoints/best.pt`; `src/export.py`
bundles the model with its preprocessing settings into `checkpoints/model.joblib`,
which both the CLI predictor and the Streamlit app load.

## Model performance

Evaluated on the held-out **test split** (60 images, stratified, class-balanced).

| Metric | Train | Validation | Test |
| --- | --- | --- | --- |
| Accuracy | 0.652 | 0.836 | **0.833** |
| ROC-AUC | 0.783 | 0.854 | **0.892** |
| PR-AUC | 0.782 | 0.826 | **0.912** |
| TB precision | 0.596 | 0.778 | **0.857** |
| TB recall | 0.957 | 0.933 | **0.800** |
| TB F1 | 0.734 | 0.849 | **0.828** |

Test confusion matrix (rows = true, columns = predicted):

```
          Pred Normal   Pred TB
Normal         26          4
TB              6         24
```

> A tiny model on a 400-image dataset has natural limits; the numbers above
> are the honest result of that trade-off, not cherry-picked peaks.

## Project structure

```
.
├── streamlit_app.py            # Streamlit UI (upload or bundled sample)
├── config.yaml                 # data, preprocessing, training settings
├── Dockerfile                  # CPU-only PyTorch + OpenCV image
├── docker-compose.yml          # one-command startup
├── requirements.txt            # pinned dependencies
├── data/
│   └── images/                 # normal/ + tuberculosis/ + metadata.csv
├── src/
│   ├── data.py                 # metadata, cleaning, splits, DataLoaders
│   ├── preprocessing.py        # OpenCV pipeline + augmentation
│   ├── model.py                # tiny CNN
│   ├── train.py                # training loop
│   ├── evaluate.py             # full metrics per split
│   ├── export.py               # joblib model artifact
│   └── predict.py              # single-image CLI inference
├── docs/
│   ├── 01_training_pipeline.ipynb
│   └── 02_image_preprocessing.ipynb
└── checkpoints/                # best.pt + model.joblib
```

## Getting started

### 1. Local (Python)

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Run the web app (model is already exported):

```bash
streamlit run streamlit_app.py
```

Open **http://localhost:8501** - either upload an X-ray or pick a bundled
sample and watch the OpenCV preview + prediction update instantly.

### 2. Docker (recommended for demo/reviewers)

```bash
docker compose up --build
```

The image ships with the trained model, sample images and the UI baked in.
Open **http://localhost:8501** - no training or extra setup needed.

## Training and evaluation from scratch

```bash
# Train (saves best checkpoint + per-epoch CSV)
python src/train.py --config config.yaml

# Evaluate the trained checkpoint on train/val/test
python src/evaluate.py --config config.yaml

# Export the joblib artifact used by the CLI and the web app
python src/export.py

# Predict a single image
python src/predict.py --model checkpoints/model.joblib data/images/tuberculosis/Tuberculosis-56.png
```

## Notebooks

- `docs/02_image_preprocessing.ipynb` - visual walkthrough of the OpenCV pipeline (CLAHE vs equalization, tensor shapes).
- `docs/01_training_pipeline.ipynb` - end-to-end: data, model, train, evaluate.

## Configuration

Everything is controlled from `config.yaml`:

```yaml
image_size: 128
preprocess:
  use_clahe: true
  clahe_clip: 1.0     # gentler than default 2.0 — higher clips hurt accuracy
train:
  batch_size: 64
  lr: 0.001
  epochs: 50
  early_stop_patience: 8
```

## Dataset

`data/images/` contains a stratified, balanced set of **400 chest X-rays**
(200 Normal + 200 Tuberculosis) from the TB Chest X-Ray database, each 256×256
grayscale, with `metadata.csv` mapping every file to its label. The raw dataset
is kept untouched under `data/raw/`.

## Limitations & future work

- 400 images is small; accuracy caps around 83% and some samples are genuinely ambiguous.
- Single-label classification only - no localization of lesions yet.
- Future: lesion bounding-box detection, Grad-CAM saliency maps in the UI, model quantization for edge deployment, and a bigger dataset.

## License

MIT © 2026 [Jayant Bhatia](LICENSE)
