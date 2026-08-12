"""
OpticBin — Frame Preprocessor
===============================
Handles frame resizing, PIL/CV2 tensor conversions, and ImageNet
normalization for the dual-model inference pipeline.
"""

import cv2
import numpy as np
import torch
from torchvision import transforms

from config.settings import INPUT_SIZE, IMAGENET_MEAN, IMAGENET_STD


# ──────────────────────────────────────────────
# Shared Transform Pipeline
# ──────────────────────────────────────────────
_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def preprocess_frame(frame: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
    """
    Accept a raw BGR OpenCV frame and return:
      - input_tensor: normalized [1, 3, 224, 224] tensor ready for inference
      - rgb_float:    [224, 224, 3] float32 array in [0, 1] for XAI overlay

    Parameters
    ----------
    frame : np.ndarray
        Raw BGR frame from cv2.VideoCapture or uploaded image.

    Returns
    -------
    tuple[torch.Tensor, np.ndarray]
        (batch tensor, RGB float image for Grad-CAM visualization)
    """
    # Convert BGR → RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Letterbox resize to target dimensions
    resized = cv2.resize(rgb, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)

    # Float [0, 1] copy for XAI heatmap blending
    rgb_float = resized.astype(np.float32) / 255.0

    # Normalize → tensor → add batch dim
    input_tensor = _transform(resized).unsqueeze(0)

    return input_tensor, rgb_float


def preprocess_pil(image) -> tuple[torch.Tensor, np.ndarray]:
    """
    Accept a PIL.Image and return (input_tensor, rgb_float).
    Convenience wrapper for uploaded images.
    """
    rgb = np.array(image.convert("RGB"))
    resized = cv2.resize(rgb, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    rgb_float = resized.astype(np.float32) / 255.0
    input_tensor = _transform(resized).unsqueeze(0)
    return input_tensor, rgb_float
