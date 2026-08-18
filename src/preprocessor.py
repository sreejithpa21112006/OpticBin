"""
OpticBin — Frame & Image Preprocessor Module
=============================================
Handles image resizing, tensor conversion, ImageNet normalization,
and image formatting for PyTorch and ONNX inference engines.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import cv2
import numpy as np
import torch
from torchvision import transforms

from config.settings import IMAGENET_MEAN, IMAGENET_STD, INPUT_SIZE

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage


class ImagePreprocessor:
    """
    Modular preprocessor for image transformation and normalization.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = INPUT_SIZE,
        mean: list[float] = IMAGENET_MEAN,
        std: list[float] = IMAGENET_STD,
    ):
        self.target_size = target_size
        self.mean = mean
        self.std = std

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    def prepare_array(self, rgb: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Convert an RGB NumPy array [H, W, 3] into:
          - input_tensor: normalized PyTorch tensor [1, 3, H, W]
          - rgb_float: float32 NumPy array in range [0, 1] for XAI rendering
        """
        resized = cv2.resize(rgb, self.target_size, interpolation=cv2.INTER_LINEAR)
        rgb_float = resized.astype(np.float32) / 255.0
        input_tensor = self.transform(resized).unsqueeze(0)
        return input_tensor, rgb_float

    def prepare_bgr_frame(self, frame: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
        """Convert BGR OpenCV frame into input tensor and normalized RGB float array."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.prepare_array(rgb)

    def prepare_pil_image(self, image: PILImage) -> Tuple[torch.Tensor, np.ndarray]:
        """Convert PIL Image into input tensor and normalized RGB float array."""
        rgb = np.array(image.convert("RGB"))
        return self.prepare_array(rgb)


# Default global preprocessor instance
_default_preprocessor = ImagePreprocessor()


def preprocess_frame(frame: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Accept a raw BGR OpenCV frame and return:
      - input_tensor: normalized [1, 3, 224, 224] tensor ready for inference
      - rgb_float:    [224, 224, 3] float32 array in [0, 1] for XAI overlay
    """
    return _default_preprocessor.prepare_bgr_frame(frame)


def preprocess_pil(image: PILImage) -> Tuple[torch.Tensor, np.ndarray]:
    """Accept a PIL.Image and return (input_tensor, rgb_float)."""
    return _default_preprocessor.prepare_pil_image(image)
