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

from config.settings import load_yaml_config
from src.trainer import ModelTrainer


def set_global_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🌱 Global seed set to {seed}")


def train_model(
    model_type: str = "efficientnetv2_s",
    data_dir: str = "dataset",
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 10,
    use_randaugment: bool = False,
    seed: int = 42,
):
    """Backward-compatible functional API for dataset training."""
    set_global_seed(seed)
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
    # Load YAML defaults for seed and data dir
    cfg = load_yaml_config()
    default_seed = cfg.get("training", {}).get("seed", 42)
    default_data = cfg.get("paths", {}).get("dataset_dir", "dataset")

    parser = argparse.ArgumentParser(description="OpticBin Model Trainer CLI")
    parser.add_argument(
        "--model",
        type=str,
        default="efficientnetv2_s",
        choices=["efficientnetv2_s", "mobilevit_xs", "all"],
        help="Backbone architecture to train (efficientnetv2_s, mobilevit_xs, or all)",
    )
    parser.add_argument("--data_dir", type=str, default=default_data, help="Dataset directory path")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (epochs)")
    parser.add_argument("--randaugment", action="store_true", help="Enable RandAugment data augmentation")
    parser.add_argument("--seed", type=int, default=default_seed, help="Global random seed for reproducibility")

    args = parser.parse_args()
    set_global_seed(args.seed)

    models_to_train = ["efficientnetv2_s", "mobilevit_xs"] if args.model == "all" else [args.model]

    for model_name in models_to_train:
        print("\n===========================================")
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
