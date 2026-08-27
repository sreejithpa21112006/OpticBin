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
    except (AttributeError, OSError):
        pass

from src.trainer import ModelTrainer


def train_model(
    model_type: str = "efficientnetv2_s",
    data_dir: str = "dataset",
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-3,
):
    """Backward-compatible functional API for dataset training."""
    trainer = ModelTrainer(
        model_type=model_type,
        data_dir=data_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
    )
    return trainer.fit()


def main() -> None:
    parser = argparse.ArgumentParser(description="OpticBin Model Trainer CLI")
    parser.add_argument(
        "--model",
        type=str,
        default="efficientnetv2_s",
        choices=["efficientnetv2_s", "mobilevit_xs", "all"],
        help="Backbone architecture to train (efficientnetv2_s, mobilevit_xs, or all)",
    )
    parser.add_argument("--data_dir", type=str, default="dataset", help="Dataset directory path")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    args = parser.parse_args()

    models_to_train = ["efficientnetv2_s", "mobilevit_xs"] if args.model == "all" else [args.model]

    for model_name in models_to_train:
        print(f"\n==========================================")
        print(f"  Training Model Architecture: {model_name}")
        print(f"==========================================\n")
        train_model(
            model_type=model_name,
            data_dir=args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )


if __name__ == "__main__":
    main()

