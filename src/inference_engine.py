"""
OpticBin — Inference Engine
=============================
Provides ONNX Runtime and PyTorch inference wrappers with automatic
hardware-accelerated execution provider selection (CUDA → CPU fallback).
"""

import time
import numpy as np
import onnxruntime as ort

from config.settings import CLASS_LABELS, NUM_CLASSES


class ONNXInferenceEngine:
    """
    High-performance ONNX Runtime inference wrapper.
    Supports INT8-quantized and FP32 ONNX models with automatic
    CUDA/CPU execution provider selection.
    """

    def __init__(self, onnx_model_path: str):
        """
        Parameters
        ----------
        onnx_model_path : str
            Path to the .onnx model file (FP32 or INT8 quantized).
        """
        # Select best available execution provider
        available = ort.get_available_providers()
        providers = []
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        self.session = ort.InferenceSession(onnx_model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        print(f"[✓] ONNX session initialized — Provider: {self.session.get_providers()[0]}")

    def predict(self, input_array: np.ndarray) -> dict:
        """
        Run forward pass and return prediction results.

        Parameters
        ----------
        input_array : np.ndarray
            Preprocessed input of shape [1, 3, 224, 224] as float32.

        Returns
        -------
        dict
            {
                "class_id": int,
                "class_label": str,
                "confidence": float,
                "probabilities": np.ndarray,
                "latency_ms": float,
            }
        """
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
