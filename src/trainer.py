"""
OpticBin — Model Trainer & Evaluator Engine
============================================
Encapsulated PyTorch fine-tuning engine supporting data augmentation,
validation tracking, learning rate scheduling, best checkpoint persistence,
and INT8 ONNX export.
"""

from __future__ import annotations

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
    ):
        self.model_type = model_type
        self.data_dir = data_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device(device)

        self.train_transforms = transforms.Compose([
            transforms.Resize(INPUT_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

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
            f"Dataset prepared: {len(train_indices)} train samples, "
            f"{len(val_indices)} val samples (stratified 80/20)."
        )
        return train_loader, val_loader

    def train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
    ) -> Tuple[float, float]:
        """Execute one training epoch and return (train_loss, train_acc)."""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(self.device), labels.to(self.device)

            optimizer.zero_grad()
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

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
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
        print(f"🚀 Initializing training pipeline for backbone: {self.model_type}")
        print(f"💻 Device: {self.device}")

        train_loader, val_loader = self.setup_dataloaders()
        model = create_backbone(self.model_type, num_classes=NUM_CLASSES, pretrained=True)
        model = model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-2)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        best_acc = 0.0
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        pt_path = os.path.join(WEIGHTS_DIR, f"{self.model_type}.pt")
        onnx_fp32 = os.path.join(WEIGHTS_DIR, f"{self.model_type}.onnx")
        onnx_int8 = os.path.join(WEIGHTS_DIR, f"{self.model_type}_int8.onnx")

        start_time = time.time()
        for epoch in range(1, self.epochs + 1):
            t_loss, t_acc = self.train_epoch(model, train_loader, criterion, optimizer)
            scheduler.step()
            v_loss, v_acc = self.validate_epoch(model, val_loader, criterion)

            print(
                f"Epoch [{epoch:02d}/{self.epochs:02d}] "
                f"Train Loss: {t_loss:.4f} | Train Acc: {t_acc*100:.2f}% | "
                f"Val Loss: {v_loss:.4f} | Val Acc: {v_acc*100:.2f}%"
            )

            if v_acc > best_acc:
                best_acc = v_acc
                torch.save(model.state_dict(), pt_path)
                print(f"  ⭐ Best checkpoint saved (Val Acc: {best_acc*100:.2f}%)")

        elapsed = time.time() - start_time
        print(f"\n🎉 Training finished in {elapsed/60:.2f} min! Best Val Acc: {best_acc*100:.2f}%")

        # Export ONNX INT8
        print("\n📦 Exporting to ONNX INT8...")
        try:
            export_to_onnx_int8(pt_path, onnx_fp32, onnx_int8, model_type=self.model_type)
        except Exception as err:
            print(f"⚠️ ONNX export warning: {err}")

        return pt_path


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
