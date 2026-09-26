"""Unit tests for configuration schemas and settings accessors."""

from __future__ import annotations

import unittest
from config.settings import (
    CLASS_LABELS,
    SUPPORTED_MODELS,
    WASTE_METADATA,
    get_app_config,
    get_bin_color,
    get_bin_icon,
    get_model_spec_obj,
    get_waste_metadata_obj,
    is_recyclable,
)


class TestConfiguration(unittest.TestCase):

    def test_class_labels_count(self):
        self.assertEqual(len(CLASS_LABELS), 3)
        self.assertIn("biodegradable", CLASS_LABELS)
        self.assertIn("non-biodegradable", CLASS_LABELS)
        self.assertIn("e-waste", CLASS_LABELS)

    def test_get_model_spec_obj(self):
        spec = get_model_spec_obj("yolov8_unified")
        self.assertEqual(spec.name, "yolov8_unified")
        self.assertEqual(spec.cam_target, "none")

        with self.assertRaises(ValueError):
            get_model_spec_obj("invalid_architecture")

    def test_get_waste_metadata_obj(self):
        meta = get_waste_metadata_obj("biodegradable")
        self.assertEqual(meta.label, "biodegradable")
        self.assertTrue(meta.biodegradable)
        self.assertEqual(meta.category, "Biodegradable / Organic")

        meta_ewaste = get_waste_metadata_obj("e-waste")
        self.assertFalse(meta_ewaste.biodegradable)

        with self.assertRaises(ValueError):
            get_waste_metadata_obj("invalid_material")

    def test_get_app_config(self):
        cfg = get_app_config()
        self.assertEqual(cfg.input_size, (224, 224))
        self.assertEqual(cfg.latency_target_ms, 100)

    def test_is_recyclable(self):
        self.assertTrue(is_recyclable("biodegradable"))
        self.assertTrue(is_recyclable("non-biodegradable"))
        self.assertTrue(is_recyclable("e-waste"))
        self.assertFalse(is_recyclable("unknown_material"))

    def test_bin_colors_and_icons(self):
        self.assertEqual(get_bin_color("biodegradable"), "#22C55E")
        self.assertEqual(get_bin_color("non-biodegradable"), "#3B82F6")
        self.assertEqual(get_bin_color("e-waste"), "#F97316")
        self.assertEqual(get_bin_icon("biodegradable"), "🟢")
        self.assertEqual(get_bin_icon("e-waste"), "🟠")

    def test_waste_metadata_keys(self):
        for label in CLASS_LABELS:
            self.assertIn(label, WASTE_METADATA)
            meta = WASTE_METADATA[label]
            self.assertIn("tips", meta)
            self.assertIn("examples", meta)
            self.assertIn("disposal", meta)
            self.assertIn("decomposition", meta)


if __name__ == "__main__":
    unittest.main()
