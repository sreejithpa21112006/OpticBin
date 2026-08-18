"""
OpticBin — Inference Benchmark Utility
======================================
Measures end-to-end inference latency (ms) and memory overhead across
PyTorch FP32, ONNX FP32, and ONNX INT8 models against the target latency budget.

Usage:
    python models/benchmark.py --iterations 100
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, Any

import numpy as np
import torch

from config.settings import LATENCY_TARGET_MS, SUPPORTED_MODELS
from src.inference_engine import ONNXInferenceEngine, PyTorchInferenceEngine
from src.model_factory import create_backbone


def benchmark_engine(
    engine_name: str,
    predict_fn: Any,
    dummy_input: Any,
    warmup: int = 10,
    iterations: int = 100,
) -> Dict[str, float]:
    """Run benchmark loop and compute mean, min, max latency statistics."""
    # Warmup runs
    for _ in range(warmup):
        predict_fn(dummy_input)

    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        predict_fn(dummy_input)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)

    latencies = np.array(latencies)
    mean_lat = float(np.mean(latencies))
    min_lat = float(np.min(latencies))
    max_lat = float(np.max(latencies))
    p95_lat = float(np.percentile(latencies, 95))

    status = "PASSED ✅" if mean_lat <= LATENCY_TARGET_MS else "EXCEEDED ⚠️"

    print(f"\n⚡ Benchmark Results: {engine_name}")
    print(f"   • Mean Latency: {mean_lat:.2f} ms ({status} vs target {LATENCY_TARGET_MS} ms)")
    print(f"   • 95th Percentile: {p95_lat:.2f} ms")
    print(f"   • Min/Max: {min_lat:.2f} ms / {max_lat:.2f} ms")

    return {
        "engine": engine_name,
        "mean_ms": mean_lat,
        "p95_ms": p95_lat,
        "min_ms": min_lat,
        "max_ms": max_lat,
        "budget_met": mean_lat <= LATENCY_TARGET_MS,
    }


def main():
    parser = argparse.ArgumentParser(description="OpticBin Model Latency Benchmark")
    parser.add_argument("--iterations", type=int, default=50, help="Number of benchmark iterations")
    parser.add_argument("--model", type=str, default="efficientnetv2_s", help="Model key to benchmark")
    args = parser.parse_args()

    print(f"🔍 Running latency benchmark for architecture: {args.model}")
    dummy_tensor = torch.randn(1, 3, 224, 224)
    dummy_numpy = dummy_tensor.numpy()

    # PyTorch Benchmark
    pytorch_model = create_backbone(args.model, pretrained=False)
    pt_engine = PyTorchInferenceEngine(pytorch_model, device="cpu")
    benchmark_engine("PyTorch FP32 (CPU)", pt_engine.predict, dummy_tensor, iterations=args.iterations)


if __name__ == "__main__":
    main()
