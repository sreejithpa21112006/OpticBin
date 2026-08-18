"""
OpticBin — Inference Engine
=============================
Provides ONNX Runtime and PyTorch inference wrappers with automatic
hardware-accelerated execution provider selection (CUDA → CPU fallback).
"""

import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from config.settings import CLASS_LABELS, WEIGHTS_DIR


def resolve_onnx_weights_path(
    model_type: str,
    weights_dir: str = WEIGHTS_DIR,
) -> Path | None:
    """Find INT8 or FP32 ONNX weights for the given model_type if available."""
    int8_path = Path(weights_dir) / f"{model_type}_int8.onnx"
    if int8_path.exists():
        return int8_path
    fp32_path = Path(weights_dir) / f"{model_type}.onnx"
    if fp32_path.exists():
        return fp32_path
    return None


class ONNXInferenceEngine:
    """
    High-performance ONNX Runtime inference wrapper.
    Supports INT8-quantized and FP32 ONNX models with automatic
    CUDA/CPU execution provider selection.
    """

    def __init__(self, onnx_model_path: str | Path, model_type: str = "efficientnetv2_s"):
        """
        Parameters
        ----------
        onnx_model_path : str | Path
            Path to the .onnx model file (FP32 or INT8 quantized).
        model_type : str
            Model architecture identifier.
        """
        self.onnx_model_path = str(onnx_model_path)
        self.model_type = model_type
        self.using_finetuned_weights = True

        available = ort.get_available_providers()
        providers = []
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        self.session = ort.InferenceSession(self.onnx_model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        print(f"[✓] ONNX session initialized ({model_type}) — Provider: {self.session.get_providers()[0]}")

    def predict(self, input_tensor_or_array: torch.Tensor | np.ndarray) -> dict:
        """
        Run forward pass and return prediction results.

        Parameters
        ----------
        input_tensor_or_array : torch.Tensor | np.ndarray
            Input tensor or float32 array of shape [1, 3, 224, 224].
        """
        if isinstance(input_tensor_or_array, torch.Tensor):
            input_array = input_tensor_or_array.detach().cpu().numpy().astype(np.float32)
        else:
            input_array = np.asarray(input_tensor_or_array, dtype=np.float32)

        start = time.perf_counter()
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_array},
        )
        latency_ms = (time.perf_counter() - start) * 1000

        logits = outputs[0][0]

        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / exp_logits.sum()

        class_id = int(np.argmax(probabilities))

        return {
            "class_id": class_id,
            "class_label": CLASS_LABELS[class_id],
            "confidence": float(probabilities[class_id]),
            "probabilities": probabilities,
            "latency_ms": round(latency_ms, 2),
        }

    def explain(
        self,
        input_tensor_or_array: torch.Tensor | np.ndarray,
        rgb_float: np.ndarray,
    ) -> dict:
        """ONNX prediction with dummy heatmap overlay for interface compatibility."""
        result = self.predict(input_tensor_or_array)
        # Produce a neutral overlay for ONNX mode when Grad-CAM is disabled
        result["heatmap_overlay"] = (rgb_float * 255).astype(np.uint8)
        return result

    def predict_and_explain(
        self,
        input_tensor_or_array: torch.Tensor | np.ndarray,
        rgb_float: np.ndarray,
    ) -> dict:
        return self.explain(input_tensor_or_array, rgb_float)

