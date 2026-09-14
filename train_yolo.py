"""
OpticBin - Unified Single-Stage YOLOv8 Waste Training Pipeline
==============================================================
Fine-tunes YOLOv8 detection models to simultaneously predict bounding boxes
and material categories (cardboard, glass, metal, paper, plastic) in a single pass.

Usage:
    python train_yolo.py --epochs 30 --batch_size 16 --imgsz 640
    python train_yolo.py --model yolov8s.pt --epochs 50
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

CONFIG_PATH = Path("dataset_roboflow/data.yaml") if Path("dataset_roboflow/data.yaml").exists() else Path("config/waste_yolo.yaml")
WEIGHTS_DEST = Path("models/weights/yolov8_waste.pt")


def train_yolo(
    data_cfg: str = str(CONFIG_PATH),
    model_name: str = "yolov8n.pt",
    epochs: int = 30,
    batch_size: int = 16,
    imgsz: int = 640,
    device: str = "",
    project: str = "results/yolo_train",
    name: str = "waste_model",
) -> None:
    """Train YOLOv8 on the waste dataset, save checkpoints, and export to ONNX."""
    print("[INFO] Initializing YOLO training pipeline...")
    print(f"       Model: {model_name}")
    print(f"       Dataset: {data_cfg}")
    print(f"       Epochs: {epochs}, Batch size: {batch_size}, Image size: {imgsz}")

    # Initialize model
    model = YOLO(model_name)

    # Determine optimal compute device
    if not device:
        import torch
        device = 0 if torch.cuda.is_available() else "cpu"

    # Train model with robust augmentations and cosine LR
    train_args = {
        "data": data_cfg,
        "epochs": epochs,
        "batch": batch_size,
        "imgsz": imgsz,
        "project": project,
        "name": name,
        "device": device,
        "workers": 2,
        "exist_ok": True,
        "verbose": True,
        "cos_lr": True,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 10.0,
        "fliplr": 0.5,
        "mosaic": 0.8,
        "mixup": 0.0,
    }

    print(f"[INFO] Compute device selected: {device}")
    print("[INFO] Starting training...")
    results = model.train(**train_args)

    # Locate best checkpoint
    candidate_paths = [
        getattr(getattr(model, "trainer", None), "best", None),
        Path("runs") / "detect" / project / name / "weights" / "best.pt",
        Path(project) / name / "weights" / "best.pt",
    ]
    best_pt = None
    for cand in candidate_paths:
        if cand and Path(cand).exists():
            best_pt = Path(cand)
            break

    if best_pt:
        WEIGHTS_DEST.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_pt, WEIGHTS_DEST)
        print(f"\n[OK] Best model checkpoint saved to: {WEIGHTS_DEST.resolve()}")

        # Export to ONNX for fast inference
        print("[INFO] Exporting trained checkpoint to ONNX...")
        try:
            trained_model = YOLO(str(WEIGHTS_DEST))
            onnx_path = trained_model.export(format="onnx")
            print(f"[OK] ONNX model exported to: {onnx_path}")
        except Exception as err:
            print(f"[WARN] ONNX export failed: {err}")
    else:
        print(f"[WARN] Training completed but best checkpoint was not found in: {candidate_paths}")

    print("\n[OK] YOLO training pipeline complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="OpticBin YOLOv8 Waste Trainer")
    parser.add_argument("--data", type=str, default=str(CONFIG_PATH), help="Path to waste_yolo.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base model (yolov8n.pt, yolov8s.pt)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Input resolution")
    parser.add_argument("--device", type=str, default="", help="Device: '0', 'cpu', 'cuda'")
    parser.add_argument("--project", type=str, default="results/yolo_train", help="Output directory")
    parser.add_argument("--name", type=str, default="waste_model", help="Run name")
    args = parser.parse_args()

    train_yolo(
        data_cfg=args.data,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
