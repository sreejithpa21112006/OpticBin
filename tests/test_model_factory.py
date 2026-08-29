"""Unit tests for model factory module."""

from __future__ import annotations

import unittest
import torch
from src.model_factory import create_backbone, get_cam_target_layers, get_model_spec


class TestModelFactory(unittest.TestCase):

    def test_get_model_spec(self):
        spec = get_model_spec("efficientnetv2_s")
        self.assertIn("timm_name", spec)
        self.assertIn("cam_target", spec)

    def test_create_backbone(self):
        model = create_backbone("efficientnetv2_s", num_classes=5, pretrained=False)
        self.assertIsNotNone(model)

        dummy_input = torch.randn(1, 3, 224, 224)
        output = model(dummy_input)
        self.assertEqual(output.shape, (1, 5))

    def test_get_cam_target_layers(self):
        model = create_backbone("efficientnetv2_s", num_classes=5, pretrained=False)
        targets = get_cam_target_layers(model, "efficientnetv2_s")
        self.assertEqual(len(targets), 1)


if __name__ == "__main__":
    unittest.main()
