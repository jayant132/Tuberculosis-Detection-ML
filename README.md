# Tuberculosis-Detection-ML

Detects tuberculosis (TB) from chest X-rays with a **tiny CNN** (~16k
parameters) that trains in seconds on a laptop CPU. Image loading and
preprocessing are done with **OpenCV**.

## How it works

1. **Preprocessing (`src/preprocessing.py`)** - every X-ray is loaded as
   grayscale with OpenCV, boosted with **CLAHE** (local contrast equalization,
   which makes faint TB lesions visible), resized to a fixed square and
   normalized to `[0, 1]`.
2. **Tiny model (`src/model.py`)** - 3 conv blocks + global average pooling.
   Small enough that it never crashes a laptop and does not overfit the
   400-image dataset.
3. **Training (`src/train.py`)** - class-weighted cross entropy, early
   stopping, best checkpoint saved to `checkpoints/best.pt`, per-epoch
   metrics to `experiments/run.csv`.
4. **Evaluation (`src/evaluate.py`)** - accuracy, ROC-AUC, PR-AUC,
   precision/recall/F1 and confusion matrices for train/val/test.
5. **Inference** - `src/export.py` saves the trained model as a **joblib**
   artifact (`checkpoints/model.joblib`), `src/predict.py` classifies a
   single image, and `app.py` wraps it all in a **Streamlit** UI.

## Setup

```bash
pip install opencv-python torch numpy pandas scikit-learn pyyaml matplotlib jupyter joblib streamlit
```

## Usage

Train and evaluate:

```bash
python src/train.py --config config.yaml
python src/evaluate.py --config config.yaml --checkpoint checkpoints/best.pt
```

Export the model with joblib and run inference on an image:

```bash
python src/export.py
python src/predict.py --model checkpoints/model.joblib path/to/xray.png
```

Launch the web UI (upload an X-ray, see the OpenCV-enhanced preview and prediction):

```bash
streamlit run app.py
```

Notebooks (in `docs/`):

- `opencv_preprocessing.ipynb` - visual walkthrough of the OpenCV pipeline.
- `walkthrough.ipynb` - full train + evaluate flow.

## Configuration

Everything lives in `config.yaml`: image size, split ratios, CLAHE settings
(`use_clahe`, `clahe_clip`, `clahe_tile`) and training hyperparameters.

## Dataset

`data/raw/TB_Chest_Radiography_Database/` - 200 normal + 200 TB chest X-rays
with a `metadata.csv` listing each image and its label.
