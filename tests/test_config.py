"""Unit tests for configuration schemas and settings accessors."""

from __future__ import annotations

import unittest
from config.settings import (
    CLASS_LABELS,
    SUPPORTED_MODELS,
    WASTE_METADATA,
    get_app_config,
    get_model_spec_obj,
    get_waste_metadata_obj,
    is_recyclable,
)


class TestConfiguration(unittest.TestCase):

    def test_class_labels_count(self):
        self.assertEqual(len(CLASS_LABELS), 5)
        self.assertIn("plastic", CLASS_LABELS)
        self.assertIn("glass", CLASS_LABELS)

    def test_get_model_spec_obj(self):
        spec = get_model_spec_obj("efficientnetv2_s")
        self.assertEqual(spec.name, "efficientnetv2_s")
        self.assertEqual(spec.cam_target, "conv_head")
        self.assertEqual(spec.timm_name, "efficientnetv2_rw_s")

        with self.assertRaises(ValueError):
            get_model_spec_obj("invalid_architecture")

    def test_get_waste_metadata_obj(self):
        meta = get_waste_metadata_obj("paper")
        self.assertEqual(meta.label, "paper")
        self.assertTrue(meta.biodegradable)
        self.assertEqual(meta.category, "Biodegradable")

        with self.assertRaises(ValueError):
            get_waste_metadata_obj("invalid_material")

    def test_get_app_config(self):
        cfg = get_app_config()
        self.assertEqual(cfg.input_size, (224, 224))
        self.assertEqual(cfg.latency_target_ms, 100)

    def test_is_recyclable(self):
        self.assertTrue(is_recyclable("paper"))
        self.assertTrue(is_recyclable("plastic"))
        self.assertFalse(is_recyclable("unknown_material"))


if __name__ == "__main__":
    unittest.main()
