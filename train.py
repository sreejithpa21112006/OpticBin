"""
OpticBin — Model Fine-Tuning & Training CLI Entrypoint
======================================================
Fine-tunes EfficientNetV2-S or MobileViT-XS on the 5-class waste dataset,
evaluates Top-1 Accuracy, saves PyTorch .pt weights, and exports to INT8 ONNX.

Usage:
    python train.py --model efficientnetv2_s --epochs 15 --batch_size 32
    python train.py --model mobilevit_xs --epochs 15 --batch_size 32
"""

import argparse
import logging
import sys
import warnings

# Suppress non-critical HuggingFace Hub unauthenticated warnings
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*HF_TOKEN*")

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.trainer import ModelTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="OpticBin Model Trainer CLI")
    parser.add_argument(
        "--model",
        type=str,
        default="efficientnetv2_s",
        choices=["efficientnetv2_s", "mobilevit_xs"],
        help="Backbone architecture to train",
    )
    parser.add_argument("--data_dir", type=str, default="dataset", help="Dataset directory path")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    args = parser.parse_args()

    trainer = ModelTrainer(
        model_type=args.model,
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
