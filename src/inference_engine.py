"""
OpticBin — Core Inference Engine Infrastructure
================================================
Provides unified abstract interface and implementations for both ONNX Runtime
and PyTorch inference engines with automatic hardware acceleration.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort

from config.settings import CLASS_LABELS, NUM_CLASSES


@dataclass(frozen=True)
class PredictionResult:
    """Standardized result container returned by all inference engines."""
    class_id: int
    class_label: str
    confidence: float
    probabilities: np.ndarray
    latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert container to legacy dict format for frontend components."""
        return {
            "class_id": self.class_id,
            "class_label": self.class_label,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "latency_ms": self.latency_ms,
        }


class BaseInferenceEngine(ABC):
    """Abstract Base Class for model inference engines."""

    @abstractmethod
    def predict(self, input_data: Any) -> PredictionResult:
        """Run forward prediction pass and return structured PredictionResult."""
        pass


class ONNXInferenceEngine(BaseInferenceEngine):
    """
    High-performance ONNX Runtime inference wrapper.
    Supports INT8-quantized and FP32 ONNX models with automatic
    CUDA/CPU execution provider selection.
    """

    def __init__(self, onnx_model_path: str):
        available = ort.get_available_providers()
        providers = []
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        self.session = ort.InferenceSession(onnx_model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.provider = self.session.get_providers()[0]

        print(f"[✓] ONNX session initialized — Provider: {self.provider}")

    def predict(self, input_array: np.ndarray) -> PredictionResult:
        """
        Run forward pass on input_array [1, 3, 224, 224] float32.
        """
        start = time.perf_counter()
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_array},
        )
        latency_ms = (time.perf_counter() - start) * 1000

        logits = outputs[0][0]

        # Softmax computation
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / exp_logits.sum()
        class_id = int(np.argmax(probabilities))

        return PredictionResult(
            class_id=class_id,
            class_label=CLASS_LABELS[class_id],
            confidence=float(probabilities[class_id]),
            probabilities=probabilities,
            latency_ms=round(latency_ms, 2),
        )


class PyTorchInferenceEngine(BaseInferenceEngine):
    """
    Native PyTorch inference engine wrapper with automatic device selection.
    """

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()

    def predict(self, input_tensor: torch.Tensor) -> PredictionResult:
        """
        Run PyTorch model forward pass.
        """
        start = time.perf_counter()
        tensor_dev = input_tensor.to(self.device)

        with torch.no_grad():
            logits = self.model(tensor_dev)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        latency_ms = (time.perf_counter() - start) * 1000
        class_id = int(np.argmax(probs))

        return PredictionResult(
            class_id=class_id,
            class_label=CLASS_LABELS[class_id],
            confidence=float(probs[class_id]),
            probabilities=probs,
            latency_ms=round(latency_ms, 2),
        )
