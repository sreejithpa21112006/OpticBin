"""
OpticBin — Frame & Image Preprocessor
======================================
Single evaluation path shared by training validation, PyTorch, and ONNX:

    RGB → PIL bilinear Resize(224, 224) → ToTensor → ImageNet normalize

The overlay image is the same 224×224 RGB in [0, 1], without normalization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import cv2
import numpy as np
import torch
from PIL import Image as PILImage
from torchvision import transforms

from config.settings import IMAGENET_MEAN, IMAGENET_STD, INPUT_SIZE

if TYPE_CHECKING:
    from PIL.Image import Image as PILImageType


def build_eval_transform(
    target_size: Tuple[int, int] = INPUT_SIZE,
    mean: list[float] | None = None,
    std: list[float] | None = None,
) -> transforms.Compose:
    """Validation / inference transform. Must stay in lockstep with ModelTrainer."""
    return transforms.Compose(
        [
            transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=mean if mean is not None else IMAGENET_MEAN,
                std=std if std is not None else IMAGENET_STD,
            ),
        ]
    )


def center_crop_square(image: "PILImageType", crop_factor: float = 0.85) -> "PILImageType":
    """Extract central square region from image to remove peripheral background noise."""
    w, h = image.size
    min_dim = min(w, h)
    crop_size = max(1, int(min_dim * crop_factor))
    left = (w - crop_size) // 2
    top = (h - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size
    return image.crop((left, top, right, bottom))


class ImagePreprocessor:
    """Convert BGR frames or PIL images into model tensors plus overlay RGB."""

    def __init__(
        self,
        target_size: Tuple[int, int] = INPUT_SIZE,
        mean: list[float] | None = None,
        std: list[float] | None = None,
        center_crop: bool = True,
        crop_factor: float = 0.85,
    ):
        self.target_size = target_size
        self.mean = mean if mean is not None else IMAGENET_MEAN
        self.std = std if std is not None else IMAGENET_STD
        self.center_crop = center_crop
        self.crop_factor = crop_factor
        self._to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )
        # Kept for callers / trainer lockstep documentation.
        self.transform = build_eval_transform(self.target_size, self.mean, self.std)

    def prepare_array(self, rgb: np.ndarray, center_crop: bool | None = None) -> Tuple[torch.Tensor, np.ndarray]:
        """RGB uint8 [H, W, 3] → (tensor [1, 3, 224, 224], float RGB [224, 224, 3])."""
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"Expected RGB array [H, W, 3], got shape {rgb.shape}.")
        return self.prepare_pil_image(
            PILImage.fromarray(np.ascontiguousarray(rgb)).convert("RGB"),
            center_crop=center_crop,
        )

    def prepare_bgr_frame(self, frame: np.ndarray, center_crop: bool | None = None) -> Tuple[torch.Tensor, np.ndarray]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.prepare_array(rgb, center_crop=center_crop)

    def prepare_pil_image(
        self,
        image: "PILImageType",
        center_crop: bool | None = None,
    ) -> Tuple[torch.Tensor, np.ndarray]:
        rgb = image.convert("RGB")
        use_crop = self.center_crop if center_crop is None else center_crop
        if use_crop:
            rgb = center_crop_square(rgb, crop_factor=self.crop_factor)

        resized = rgb.resize(self.target_size, PILImage.BILINEAR)
        rgb_float = np.clip(np.asarray(resized, dtype=np.float32) / 255.0, 0.0, 1.0)
        input_tensor = self._to_tensor(resized).unsqueeze(0)
        return input_tensor, rgb_float


_default_preprocessor = ImagePreprocessor(center_crop=True, crop_factor=0.85)


def preprocess_frame(frame: np.ndarray, center_crop: bool = True) -> Tuple[torch.Tensor, np.ndarray]:
    """BGR OpenCV frame → (input_tensor, rgb_float)."""
    return _default_preprocessor.prepare_bgr_frame(frame, center_crop=center_crop)


def preprocess_pil(image: "PILImageType", center_crop: bool = True) -> Tuple[torch.Tensor, np.ndarray]:
    """PIL image → (input_tensor, rgb_float)."""
    return _default_preprocessor.prepare_pil_image(image, center_crop=center_crop)


ImagePreprocessor = ImagePreprocessor
build_eval_transform = build_eval_transform
center_crop_square = center_crop_square

