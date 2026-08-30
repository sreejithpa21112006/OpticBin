"""
OpticBin — Model Fine-Tuning & Training CLI Entrypoint
======================================================
Fine-tunes EfficientNetV2-S or MobileViT-XS on the 5-class waste dataset,
evaluates Top-1 Accuracy, saves PyTorch .pt weights, and exports to INT8 ONNX.

Usage:
    python train.py --model efficientnetv2_s --epochs 15 --batch_size 32
    python train.py --model mobilevit_xs --epochs 15 --batch_size 32
    python train.py --model all --epochs 15 --batch_size 32
    python train.py --seed 42  # Set random seed for reproducibility
"""

import argparse
import logging
import os
import random
import sys
import warnings

import numpy as np
import torch

# Suppress non-critical HuggingFace Hub unauthenticated warnings
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*HF_TOKEN*")

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

from config.settings import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_PATIENCE,
    DEFAULT_SEED,
    DATASET_DIR,
)
from src.trainer import ModelTrainer


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"🎲 Random seed set to {seed} for reproducibility.", flush=True)


def train_model(
    model_type: str = "efficientnetv2_s",
    data_dir: str = DATASET_DIR,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LEARNING_RATE,
    patience: int = DEFAULT_PATIENCE,
    use_randaugment: bool = False,
    seed: int = DEFAULT_SEED,
):
    """Backward-compatible functional API for dataset training."""
    set_seed(seed)
    trainer = ModelTrainer(
        model_type=model_type,
        data_dir=data_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        use_randaugment=use_randaugment,
        seed=seed,
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
    parser.add_argument("--data_dir", type=str, default=DATASET_DIR, help="Dataset directory path")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE, help="Learning rate")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE, help="Early stopping patience (epochs)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducibility")
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
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
