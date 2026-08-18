"""
OpticBin — Comprehensive Unit & Integration Tests
"""

import os
import shutil
import tempfile
import unittest

import numpy as np
import torch
from PIL import Image

from config.settings import CLASS_LABELS, NUM_CLASSES, SUPPORTED_MODELS
from download_dataset import generate_synthetic_dataset, print_dataset_info
from src.inference_engine import resolve_onnx_weights_path
from src.model_factory import (
    create_backbone,
    get_cam_target_layers,
    infer_model_type_from_path,
    load_checkpoint,
)
from src.preprocessor import preprocess_frame, preprocess_pil
from src.xai_engine import RecyclingXAIEngine
from ui.components import render_engine_status


class TestONNXEngine(unittest.TestCase):
    def test_resolve_onnx_weights_path_none(self):
        result = resolve_onnx_weights_path("non_existent_model")
        self.assertIsNone(result)

    def test_export_and_onnx_engine(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = create_backbone("efficientnetv2_s", num_classes=NUM_CLASSES, pretrained=False)
            pt_path = os.path.join(tmpdir, "model.pt")
            onnx_path = os.path.join(tmpdir, "model.onnx")
            quant_path = os.path.join(tmpdir, "model_int8.onnx")

            torch.save(model.state_dict(), pt_path)

            from models.export_onnx import export_to_onnx_int8
            export_to_onnx_int8(pt_path, onnx_path, quant_path, model_type="efficientnetv2_s")

            from src.inference_engine import ONNXInferenceEngine
            onnx_engine = ONNXInferenceEngine(quant_path, model_type="efficientnetv2_s")

            dummy_tensor = torch.randn(1, 3, 224, 224)
            rgb_float = np.ones((224, 224, 3), dtype=np.float32)

            pred = onnx_engine.predict(dummy_tensor)
            self.assertIn("class_label", pred)
            self.assertIn(pred["class_label"], CLASS_LABELS)

            explain_res = onnx_engine.explain(dummy_tensor, rgb_float)
            self.assertIn("heatmap_overlay", explain_res)


class TestModelFactory(unittest.TestCase):
    def test_supported_models_spec(self):
        self.assertIn("efficientnetv2_s", SUPPORTED_MODELS)
        self.assertIn("mobilevit_xs", SUPPORTED_MODELS)

    def test_create_backbone(self):
        model = create_backbone("efficientnetv2_s", num_classes=NUM_CLASSES, pretrained=False)
        self.assertIsNotNone(model)
        target_layers = get_cam_target_layers(model, "efficientnetv2_s")
        self.assertEqual(len(target_layers), 1)

    def test_infer_model_type_from_path(self):
        self.assertEqual(infer_model_type_from_path("models/weights/efficientnetv2_s.pt"), "efficientnetv2_s")
        self.assertEqual(infer_model_type_from_path("models/weights/mobilevit_xs_int8.onnx"), "mobilevit_xs")

    def test_load_checkpoint_with_module_prefix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = create_backbone("efficientnetv2_s", num_classes=NUM_CLASSES, pretrained=False)
            state_dict = model.state_dict()
            module_state_dict = {f"module.{k}": v for k, v in state_dict.items()}

            pt_path = os.path.join(tmpdir, "module_model.pt")
            torch.save(module_state_dict, pt_path)

            loaded_model = load_checkpoint(model, pt_path)
            self.assertIsNotNone(loaded_model)


class TestUIComponents(unittest.TestCase):
    def test_render_engine_status(self):
        pytorch_engine = RecyclingXAIEngine("efficientnetv2_s", weights_path=None)
        # Verify render_engine_status handles PyTorch and fallback states cleanly
        render_engine_status("efficientnetv2_s", "PyTorch + Grad-CAM", pytorch_engine)
        render_engine_status("efficientnetv2_s", "ONNX Runtime (Fast)", pytorch_engine)


class TestPreprocessor(unittest.TestCase):
    def test_preprocess_frame(self):
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        input_tensor, rgb_float = preprocess_frame(dummy_frame)
        self.assertEqual(input_tensor.shape, (1, 3, 224, 224))
        self.assertEqual(rgb_float.shape, (224, 224, 3))
        self.assertTrue(0.0 <= rgb_float.min() and rgb_float.max() <= 1.0)

    def test_preprocess_pil(self):
        pil_img = Image.new("RGB", (300, 300), color=(100, 150, 200))
        input_tensor, rgb_float = preprocess_pil(pil_img)
        self.assertEqual(input_tensor.shape, (1, 3, 224, 224))
        self.assertEqual(rgb_float.shape, (224, 224, 3))


class TestXAIEngine(unittest.TestCase):
    def test_recycling_xai_engine_predict_and_explain(self):
        engine = RecyclingXAIEngine("efficientnetv2_s", weights_path=None)
        input_tensor = torch.randn(1, 3, 224, 224)
        rgb_float = np.ones((224, 224, 3), dtype=np.float32)

        pred = engine.predict(input_tensor)
        self.assertIn("class_label", pred)
        self.assertIn(pred["class_label"], CLASS_LABELS)
        self.assertEqual(len(pred["probabilities"]), NUM_CLASSES)

        explain_res = engine.explain(input_tensor, rgb_float)
        self.assertIn("heatmap_overlay", explain_res)
        self.assertEqual(explain_res["heatmap_overlay"].shape, (224, 224, 3))


class TestDatasetUtils(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_generate_synthetic_dataset(self):
        generate_synthetic_dataset(dest_dir=self.test_dir, samples_per_class=2)
        for label in CLASS_LABELS:
            cls_dir = os.path.join(self.test_dir, label)
            self.assertTrue(os.path.exists(cls_dir))
            files = os.listdir(cls_dir)
            self.assertEqual(len(files), 2)

        # Ensure print_dataset_info runs cleanly
        print_dataset_info(self.test_dir)


if __name__ == "__main__":
    unittest.main()
