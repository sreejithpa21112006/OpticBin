"""
OpticBin — Model Fine-Tuning & Training CLI Entrypoint
======================================================
Fine-tunes EfficientNetV2-S or MobileViT-XS on the 5-class waste dataset,
evaluates Top-1 Accuracy, saves PyTorch .pt weights, and exports to INT8 ONNX.

Usage:
    python train.py --model efficientnetv2_s --epochs 15 --batch_size 32
    python train.py --model mobilevit_xs --epochs 15 --batch_size 32
    python train.py --model all --epochs 15 --batch_size 32
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
    patience: int = 10,
    use_randaugment: bool = False,
):
    """Backward-compatible functional API for dataset training."""
    trainer = ModelTrainer(
        model_type=model_type,
        data_dir=data_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        use_randaugment=use_randaugment,
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
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (epochs)")
    parser.add_argument("--randaugment", action="store_true", help="Enable RandAugment data augmentation")

    args = parser.parse_args()

    models_to_train = ["efficientnetv2_s", "mobilevit_xs"] if args.model == "all" else [args.model]

    for model_name in models_to_train:
        print("\n==========================================")
        print(f"  Training Model Architecture: {model_name}")
        print("==========================================\n")
        train_model(
            model_type=model_name,
            data_dir=args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            use_randaugment=args.randaugment,
        )


if __name__ == "__main__":
    main()
