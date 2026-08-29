"""
OpticBin — Inference Benchmark Utility
======================================
Measures preprocess + inference latency and optional RSS memory for
PyTorch, Grad-CAM, ONNX FP32, and ONNX INT8.

Usage:
    python models/benchmark.py --iterations 50 --model efficientnetv2_s
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.settings import LATENCY_TARGET_MS, WEIGHTS_DIR
from src.inference_engine import ONNXInferenceEngine, PyTorchInferenceEngine
from src.model_factory import create_backbone, resolve_weights_path
from src.preprocessor import ImagePreprocessor
from src.xai_engine import RecyclingXAIEngine


def _rss_mb() -> float | None:
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return None


def benchmark_fn(
    name: str,
    fn: Callable[[], Any],
    warmup: int = 8,
    iterations: int = 50,
) -> dict[str, Any]:
    for _ in range(warmup):
        fn()

    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        latencies.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(latencies)
    mean_lat = float(np.mean(arr))
    status = "WITHIN TARGET" if mean_lat <= LATENCY_TARGET_MS else "OVER TARGET"
    rss = _rss_mb()
    rss_txt = f"{rss:.0f} MB RSS" if rss is not None else "RSS n/a (install psutil)"

    print(f"\n{name}")
    print(f"   mean {mean_lat:.2f} ms  p95 {float(np.percentile(arr, 95)):.2f} ms  "
          f"min {float(np.min(arr)):.2f} / max {float(np.max(arr)):.2f}  [{status}]  {rss_txt}")

    return {
        "name": name,
        "mean_ms": mean_lat,
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "within_target": mean_lat <= LATENCY_TARGET_MS,
        "rss_mb": rss,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="OpticBin latency benchmark")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--model", type=str, default="efficientnetv2_s")
    args = parser.parse_args()

    print(f"Benchmarking '{args.model}' against {LATENCY_TARGET_MS:.0f} ms target")
    print("Note: dummy ImageNet-head models are used if no fine-tuned weights exist.")

    preprocessor = ImagePreprocessor()
    dummy_pil = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
    dummy_tensor, rgb_float = preprocessor.prepare_pil_image(dummy_pil)

    benchmark_fn(
        "Preprocess 480x640 RGB (PIL resize + normalize)",
        lambda: preprocessor.prepare_pil_image(dummy_pil),
        iterations=args.iterations,
    )

    pt_model = create_backbone(args.model, pretrained=False)
    weights = resolve_weights_path(args.model)
    if weights:
        from src.model_factory import load_checkpoint

        pt_model = load_checkpoint(pt_model, weights)
        print(f"Loaded fine-tuned weights: {weights}")
    pt_engine = PyTorchInferenceEngine(pt_model, device="cpu")
    benchmark_fn(
        "PyTorch predict (CPU, classification only)",
        lambda: pt_engine.predict(dummy_tensor),
        iterations=args.iterations,
    )

    xai = RecyclingXAIEngine(args.model, weights_path=str(weights) if weights else None)
    benchmark_fn(
        "PyTorch + Grad-CAM (CPU, explain)",
        lambda: xai.explain(dummy_tensor, rgb_float),
        warmup=3,
        iterations=max(8, args.iterations // 5),
    )

    weights_dir = Path(WEIGHTS_DIR)
    for tag, filename in (
        ("ONNX FP32", f"{args.model}.onnx"),
        ("ONNX INT8", f"{args.model}_int8.onnx"),
    ):
        path = weights_dir / filename
        if not path.exists():
            print(f"\n{tag}: skipped (missing {path})")
            continue
        onnx_engine = ONNXInferenceEngine(path, model_type=args.model)
        benchmark_fn(
            f"{tag} predict (CPU)",
            lambda engine=onnx_engine: engine.predict(dummy_tensor),
            iterations=args.iterations,
        )


if __name__ == "__main__":
    main()
