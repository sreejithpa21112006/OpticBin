"""
OpticBin — Model Trainer & Evaluator Engine
============================================
Encapsulated PyTorch fine-tuning engine supporting data augmentation,
validation tracking, learning rate scheduling, best checkpoint persistence,
and INT8 ONNX export.
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from config.settings import (
    DEVICE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_SIZE,
    NUM_CLASSES,
    WEIGHTS_DIR,
)
from models.export_onnx import export_to_onnx_int8
from src.model_factory import create_backbone
from src.preprocessor import build_eval_transform


def _stratified_split(
    targets: List[int],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """Split indices per class so train and val keep the same class mix."""
    rng = torch.Generator().manual_seed(seed)
    by_class: dict[int, List[int]] = defaultdict(list)
    for index, label in enumerate(targets):
        by_class[int(label)].append(index)

    train_indices: List[int] = []
    val_indices: List[int] = []
    for indices in by_class.values():
        perm = torch.randperm(len(indices), generator=rng).tolist()
        shuffled = [indices[i] for i in perm]
        n_val = max(1, int(round(len(shuffled) * val_ratio))) if len(shuffled) > 1 else 0
        val_indices.extend(shuffled[:n_val])
        train_indices.extend(shuffled[n_val:])
    return train_indices, val_indices


class ModelTrainer:
    """
    Encapsulates dataset preparation, model training, evaluation,
    checkpoint saving, and ONNX export.
    """

    def __init__(
        self,
        model_type: str = "efficientnetv2_s",
        data_dir: str = "dataset",
        epochs: int = 15,
        batch_size: int = 32,
        lr: float = 1e-3,
        device: str = DEVICE,
        patience: int = 10,
        use_randaugment: bool = False,
    ):
        self.model_type = model_type
        self.data_dir = data_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device(device)
        self.patience = patience
        self.use_randaugment = use_randaugment

        train_transform_list = [
            transforms.Resize(INPUT_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
        ]
        if self.use_randaugment and hasattr(transforms, "RandAugment"):
            train_transform_list.append(transforms.RandAugment(num_ops=2, magnitude=9))
        else:
            train_transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomRotation(degrees=20),
                transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
            ])

        train_transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        self.train_transforms = transforms.Compose(train_transform_list)
        self.val_transforms = build_eval_transform()

    def setup_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Load dataset and return (train_loader, val_loader)."""
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(
                f"Dataset directory '{self.data_dir}' not found! "
                "Please run 'python download_dataset.py' first."
            )

        train_dataset = datasets.ImageFolder(root=self.data_dir, transform=self.train_transforms)
        val_dataset = datasets.ImageFolder(root=self.data_dir, transform=self.val_transforms)

        targets = [label for _, label in train_dataset.samples]
        train_indices, val_indices = _stratified_split(targets, val_ratio=0.2, seed=42)

        train_ds = Subset(train_dataset, train_indices)
        val_ds = Subset(val_dataset, val_indices)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, num_workers=0)

        print(
            f"📊 Dataset prepared: {len(train_indices)} train samples, "
            f"{len(val_indices)} val samples (stratified 80/20).",
            flush=True,
        )
        return train_loader, val_loader

    def train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scaler: torch.amp.GradScaler | None = None,
    ) -> Tuple[float, float]:
        """Execute one training epoch with optional Automatic Mixed Precision (AMP)."""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        use_amp = self.device.type == "cuda"

        for images, labels in train_loader:
            images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if use_amp and scaler is not None:
                with torch.amp.autocast("cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total if total > 0 else 0.0
        epoch_acc = correct / total if total > 0 else 0.0
        return epoch_loss, epoch_acc

    def validate_epoch(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        criterion: nn.Module,
    ) -> Tuple[float, float]:
        """Evaluate model on validation loader and return (val_loss, val_acc)."""
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        use_amp = self.device.type == "cuda"

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                if use_amp:
                    with torch.amp.autocast("cuda"):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss = running_loss / total if total > 0 else 0.0
        val_acc = correct / total if total > 0 else 0.0
        return val_loss, val_acc

    def fit(self) -> str:
        """Run complete training, evaluation, and export loop."""
        print(f"🚀 Initializing training pipeline for backbone: {self.model_type}", flush=True)
        print(f"💻 Device: {self.device}", flush=True)
        print(f"⚙️ Hyperparams: epochs={self.epochs}, batch_size={self.batch_size}, lr={self.lr}, patience={self.patience}", flush=True)

        train_loader, val_loader = self.setup_dataloaders()
        model = create_backbone(self.model_type, num_classes=NUM_CLASSES, pretrained=True)

        # Optimize for CPU fine-tuning by freezing early backbone stem if on CPU
        if self.device.type == "cpu":
            print("⚡ CPU fine-tuning acceleration: freezing early backbone layers.", flush=True)
            for name, param in model.named_parameters():
                if not any(k in name for k in ["classifier", "head", "conv_head", "final_conv", "blocks.5", "blocks.6", "stages.3"]):
                    param.requires_grad = False

        model = model.to(self.device)

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(trainable_params, lr=self.lr, weight_decay=1e-2)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-6)

        best_acc = 0.0
        best_loss = float("inf")
        patience_counter = 0

        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        pt_path = os.path.join(WEIGHTS_DIR, f"{self.model_type}.pt")
        onnx_fp32 = os.path.join(WEIGHTS_DIR, f"{self.model_type}.onnx")
        onnx_int8 = os.path.join(WEIGHTS_DIR, f"{self.model_type}_int8.onnx")
        metrics_path = os.path.join(WEIGHTS_DIR, f"{self.model_type}_metrics.json")

        history = []
        scaler = torch.amp.GradScaler("cuda") if self.device.type == "cuda" else None
        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            t_loss, t_acc = self.train_epoch(model, train_loader, criterion, optimizer, scaler=scaler)
            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()
            v_loss, v_acc = self.validate_epoch(model, val_loader, criterion)

            epoch_record = {
                "epoch": epoch,
                "train_loss": round(t_loss, 4),
                "train_acc": round(t_acc, 4),
                "val_loss": round(v_loss, 4),
                "val_acc": round(v_acc, 4),
                "lr": current_lr,
            }
            history.append(epoch_record)

            print(
                f"Epoch [{epoch:02d}/{self.epochs:02d}] "
                f"Train Loss: {t_loss:.4f} | Train Acc: {t_acc*100:.2f}% | "
                f"Val Loss: {v_loss:.4f} | Val Acc: {v_acc*100:.2f}% | LR: {current_lr:.6f}",
                flush=True,
            )

            if v_acc > best_acc or (abs(v_acc - best_acc) < 1e-4 and v_loss < best_loss):
                best_acc = v_acc
                best_loss = v_loss
                patience_counter = 0
                torch.save(model.state_dict(), pt_path)
                print(f"  ⭐ Best checkpoint saved (Val Acc: {best_acc*100:.2f}%)", flush=True)
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"🛑 Early stopping triggered after {epoch} epochs (no improvement for {self.patience} epochs).")
                    break

        elapsed = time.time() - start_time
        print(f"\n🎉 Training finished in {elapsed/60:.2f} min! Best Val Acc: {best_acc*100:.2f}%", flush=True)

        # Save training metrics to JSON
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({
                "model_type": self.model_type,
                "epochs_completed": len(history),
                "best_val_acc": round(best_acc, 4),
                "total_time_seconds": round(elapsed, 2),
                "history": history,
            }, f, indent=2)
        print(f"📊 Training metrics saved to '{metrics_path}'")

        # Export ONNX INT8
        print("\n📦 Exporting to ONNX INT8...", flush=True)
        try:
            export_to_onnx_int8(pt_path, onnx_fp32, onnx_int8, model_type=self.model_type)
        except Exception as err:
            print(f"⚠️ ONNX export warning: {err}", flush=True)

        return pt_path
