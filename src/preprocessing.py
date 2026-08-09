"""OpenCV image preprocessing for chest X-rays.

Each step is one small function so the pipeline stays readable and easy
to tweak. Chest X-rays are grayscale, low contrast and shot at different
sizes, so we: load grayscale -> enhance local contrast (CLAHE) -> resize
to a fixed square -> convert to a torch tensor in [0, 1].
"""

import os

import cv2
import numpy as np
import torch


def read_gray(path):
    """Load an image as a single-channel uint8 array (0-255)."""
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def is_readable(path):
    """True when the file exists and OpenCV can actually decode it."""
    return os.path.isfile(path) and cv2.imread(path, cv2.IMREAD_GRAYSCALE) is not None


def apply_clahe(image, clip=2.0, tile=8):
    """Equalize local contrast so faint TB lesions become more visible."""
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    return clahe.apply(image)


def resize(image, size):
    """Downscale to a fixed square while keeping the whole lung in frame."""
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def to_tensor(image):
    """Convert uint8 pixels to a float torch tensor in [0, 1] (CxHxW)."""
    array = image.astype(np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def random_augment(image, max_rotation=10):
    """Random OpenCV augmentation for training: flip, rotate, jitter."""
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
    angle = np.random.uniform(-max_rotation, max_rotation)
    h, w = image.shape
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    image = cv2.warpAffine(
        image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )
    alpha = np.random.uniform(0.9, 1.1)
    beta = np.random.uniform(-10, 10)
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return image


def prepare_image(image, size, use_clahe=True, clip=1.0, tile=8, augment=False):
    """Run enhance + resize + normalize on an already-loaded grayscale image."""
    if augment:
        image = random_augment(image)
    if use_clahe:
        image = apply_clahe(image, clip, tile)
    image = resize(image, size)
    return to_tensor(image)


def prepare(path, size, use_clahe=True, clip=1.0, tile=8, augment=False):
    """Full pipeline: load -> enhance -> resize -> tensor."""
    return prepare_image(read_gray(path), size, use_clahe, clip, tile, augment)
