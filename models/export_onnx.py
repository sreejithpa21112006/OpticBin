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

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# torch's ONNX exporter logs non-ASCII status glyphs that crash cp1252 consoles.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.model_factory import create_model_from_checkpoint  # noqa: E402


def _consolidate_for_quantization(onnx_path: str) -> None:
    """
    Rewrite the exported graph so ONNX Runtime can quantize it.

    The exporter leaves stale intermediate `value_info` shapes that fail
    quantization shape inference, and may spill weights into a sidecar
    data file. Re-saving without them yields a self-contained model.
    """
    model_proto = onnx.load(onnx_path)
    del model_proto.graph.value_info[:]
    onnx.checker.check_model(model_proto)
    onnx.save(model_proto, onnx_path)


def export_to_onnx_int8(
    pt_model_path: str,
    output_onnx_path: str,
    quantized_onnx_path: str,
    model_type: str | None = None,
    opset_version: int = 18,
) -> None:
    """
    Two-stage export pipeline:
      1. Rebuild the backbone, load a state_dict, export FP32 ONNX
      2. Apply dynamic INT8 weight quantization via ONNX Runtime
    """
    model, resolved_type = create_model_from_checkpoint(
        pt_model_path,
        model_type=model_type,
        map_location="cpu",
    )
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        dummy_input,
        output_onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    _consolidate_for_quantization(output_onnx_path)
    fp32_size_mb = Path(output_onnx_path).stat().st_size / (1024 * 1024)
    print(f"[✓] FP32 ONNX exported ({resolved_type}) → {output_onnx_path} ({fp32_size_mb:.2f} MB)")

    quantize_dynamic(
        model_input=output_onnx_path,
        model_output=quantized_onnx_path,
        weight_type=QuantType.QUInt8,
    )
    int8_size_mb = Path(quantized_onnx_path).stat().st_size / (1024 * 1024)
    print(f"[✓] INT8 quantized ONNX exported → {quantized_onnx_path} ({int8_size_mb:.2f} MB)")
    print("Export and Quantization successfully completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpticBin ONNX INT8 Exporter")
    parser.add_argument("--pt_path", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--onnx_out", required=True, help="FP32 ONNX output path")
    parser.add_argument("--quant_out", required=True, help="INT8 ONNX output path")
    parser.add_argument(
        "--model",
        default=None,
        help="Backbone key (inferred from filename when omitted)",
    )
    args = parser.parse_args()

    export_to_onnx_int8(args.pt_path, args.onnx_out, args.quant_out, model_type=args.model)
