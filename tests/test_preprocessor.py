"""Unit tests for image preprocessor module."""

from __future__ import annotations

import unittest
import numpy as np
from PIL import Image

from src.preprocessor import ImagePreprocessor, preprocess_frame, preprocess_pil


class TestPreprocessor(unittest.TestCase):

    def test_prepare_bgr_frame(self):
        # Create synthetic BGR frame [480, 640, 3] uint8
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tensor, rgb_float = preprocess_frame(dummy_frame)

        self.assertEqual(tensor.shape, (1, 3, 224, 224))
        self.assertEqual(rgb_float.shape, (224, 224, 3))
        self.assertTrue(0.0 <= rgb_float.min() <= rgb_float.max() <= 1.0)

    def test_prepare_pil_image(self):
        # Create synthetic PIL Image
        pil_img = Image.new("RGB", (300, 300), color="red")
        tensor, rgb_float = preprocess_pil(pil_img)

        self.assertEqual(tensor.shape, (1, 3, 224, 224))
        self.assertEqual(rgb_float.shape, (224, 224, 3))

    def test_center_crop_square(self):
        pil_img = Image.new("RGB", (640, 480), color="blue")
        tensor, rgb_float = preprocess_pil(pil_img, center_crop=True)
        self.assertEqual(tensor.shape, (1, 3, 224, 224))


if __name__ == "__main__":
    unittest.main()

