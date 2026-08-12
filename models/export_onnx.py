"""
OpticBin — ONNX Dynamic INT8 Exporter
=======================================
Converts a PyTorch `.pt` checkpoint to ONNX format, then applies
dynamic INT8 quantization for edge-optimized inference.

Usage:
    python models/export_onnx.py \
        --pt_path  models/weights/efficientnetv2_s.pt \
        --onnx_out models/weights/efficientnetv2_s.onnx \
        --quant_out models/weights/efficientnetv2_s_int8.onnx
"""

import argparse
import torch
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType


def export_to_onnx_int8(
    pt_model_path: str,
    output_onnx_path: str,
    quantized_onnx_path: str,
) -> None:
    """
    Two-stage export pipeline:
      1. Export FP32 PyTorch model → ONNX graph (opset 14, dynamic batch)
      2. Apply dynamic INT8 weight quantization via ONNX Runtime
    """
    # Load PyTorch checkpoint
    model = torch.load(pt_model_path, map_location="cpu")
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)

    # ── Stage 1: Export FP32 ONNX ──
    torch.onnx.export(
        model,
        dummy_input,
        output_onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input":  {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    print(f"[✓] FP32 ONNX exported → {output_onnx_path}")

    # ── Stage 2: Apply Dynamic INT8 Quantization ──
    quantize_dynamic(
        model_input=output_onnx_path,
        model_output=quantized_onnx_path,
        weight_type=QuantType.QUInt8,
    )
    print(f"[✓] INT8 quantized ONNX exported → {quantized_onnx_path}")
    print("Export and Quantization successfully completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpticBin ONNX INT8 Exporter")
    parser.add_argument("--pt_path",   required=True, help="Path to .pt checkpoint")
    parser.add_argument("--onnx_out",  required=True, help="FP32 ONNX output path")
    parser.add_argument("--quant_out", required=True, help="INT8 ONNX output path")
    args = parser.parse_args()

    export_to_onnx_int8(args.pt_path, args.onnx_out, args.quant_out)
