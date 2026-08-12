"""
OpticBin — Model Fine-Tuning & Training Pipeline
=================================================
Fine-tunes EfficientNetV2-S or MobileViT-XS on the 5-class waste dataset,
evaluates Top-1 Accuracy, saves PyTorch .pt weights, and exports to INT8 ONNX.

Usage:
    python train.py --model efficientnetv2_s --epochs 15 --batch_size 32
    python train.py --model mobilevit_xs --epochs 15 --batch_size 32
"""

import argparse
import logging
import os
import sys
import time
import warnings

# Suppress non-critical HuggingFace Hub unauthenticated warnings
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*HF_TOKEN*")

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import timm

from config.settings import (
    CLASS_LABELS,
    NUM_CLASSES,
    INPUT_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    DEVICE,
    WEIGHTS_DIR,
)
from models.export_onnx import export_to_onnx_int8


# ──────────────────────────────────────────────
# Data Augmentations
# ──────────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_transforms = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def train_model(
    model_type: str = "efficientnetv2_s",
    data_dir: str = "dataset",
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-3,
):
    print(f"🚀 Initializing training pipeline for model: {model_type}")
    print(f"💻 Device: {DEVICE}")

    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Dataset directory '{data_dir}' not found! "
            f"Please run 'python download_dataset.py' or populate '{data_dir}/' with class subfolders."
        )

    # ── 1. Load Dataset ──
    train_dataset = datasets.ImageFolder(root=data_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(root=data_dir, transform=val_transforms)

    print(f"📊 Total images found: {len(train_dataset)} across {len(train_dataset.classes)} classes.")
    print(f"🏷️ Class mapping: {train_dataset.class_to_idx}")

    val_size = int(0.2 * len(train_dataset))
    train_size = len(train_dataset) - val_size

    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(train_dataset), generator=generator).tolist()

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_ds = torch.utils.data.Subset(train_dataset, train_indices)
    val_ds = torch.utils.data.Subset(val_dataset, val_indices)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # ── 2. Create Backbone Model ──
    timm_name = "efficientnetv2_rw_m" if model_type == "efficientnetv2_s" else "mobilevit_xs"
    model = timm.create_model(timm_name, pretrained=True, num_classes=NUM_CLASSES)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    pt_checkpoint_path = os.path.join(WEIGHTS_DIR, f"{model_type}.pt")
    onnx_fp32_path = os.path.join(WEIGHTS_DIR, f"{model_type}.onnx")
    onnx_int8_path = os.path.join(WEIGHTS_DIR, f"{model_type}_int8.onnx")

    # ── 3. Training Loop ──
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total if val_total > 0 else 0
        val_loss = val_loss / val_total if val_total > 0 else 0

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), pt_checkpoint_path)
            print(f"  ⭐ Saved best model checkpoint (Val Acc: {best_acc*100:.2f}%)")

    total_time = time.time() - start_time
    print(f"\n🎉 Training complete in {total_time/60:.2f} minutes! Best Val Accuracy: {best_acc*100:.2f}%")

    # ── 4. Export to INT8 ONNX ──
    print("\n📦 Exporting to ONNX dynamic INT8 format...")
    try:
        export_to_onnx_int8(pt_checkpoint_path, onnx_fp32_path, onnx_int8_path)
        print("✅ Model export & quantization pipeline complete!")
    except Exception as e:
        print(f"⚠️ ONNX Export note: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpticBin Model Trainer")
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
    train_model(
        model_type=args.model,
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
