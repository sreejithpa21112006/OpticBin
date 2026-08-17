"""
OpticBin — Frame Preprocessor
===============================
Handles frame resizing, PIL/CV2 tensor conversions, and ImageNet
normalization for the dual-model inference pipeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch
from torchvision import transforms

from config.settings import IMAGENET_MEAN, IMAGENET_STD, INPUT_SIZE

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage


_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def _to_inference_tensors(rgb: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
    """Resize an RGB array into a model tensor and a Grad-CAM overlay image."""
    resized = cv2.resize(rgb, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    rgb_float = resized.astype(np.float32) / 255.0
    input_tensor = _transform(resized).unsqueeze(0)
    return input_tensor, rgb_float


def preprocess_frame(frame: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
    """
    Accept a raw BGR OpenCV frame and return:
      - input_tensor: normalized [1, 3, 224, 224] tensor ready for inference
      - rgb_float:    [224, 224, 3] float32 array in [0, 1] for XAI overlay
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return _to_inference_tensors(rgb)


def preprocess_pil(image: PILImage) -> tuple[torch.Tensor, np.ndarray]:
    """Accept a PIL.Image and return (input_tensor, rgb_float)."""
    rgb = np.array(image.convert("RGB"))
    return _to_inference_tensors(rgb)
