"""
OpticBin — Model Evaluation Script
====================================
Evaluates a trained OpticBin model on a held-out test split and saves metrics
(accuracy, per-class F1, confusion matrix) to the results/ directory.

Usage:
    python evaluate.py --model efficientnetv2_s --data_dir dataset
    python evaluate.py --model mobilevit_xs --data_dir dataset --seed 42
    python evaluate.py --model all --data_dir dataset
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.settings import (
    CLASS_LABELS,
    DEVICE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_SIZE,
    NUM_CLASSES,
    WEIGHTS_DIR,
    load_yaml_config,
)
from src.model_factory import create_backbone, load_checkpoint, resolve_weights_path
from src.preprocessor import build_eval_transform


# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility Helpers
# ──────────────────────────────────────────────────────────────────────────────

def set_global_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🌱 Global seed set to {seed}")


# ──────────────────────────────────────────────────────────────────────────────
# Stratified Test Split
# ──────────────────────────────────────────────────────────────────────────────

def _stratified_test_split(
    targets: list[int],
    test_ratio: float = 0.15,
    seed: int = 42,
) -> list[int]:
    """Return indices of the held-out test portion (stratified by class)."""
    rng = torch.Generator().manual_seed(seed)
    from collections import defaultdict
    by_class: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(targets):
        by_class[int(label)].append(idx)

    test_indices: list[int] = []
    for indices in by_class.values():
        perm = torch.randperm(len(indices), generator=rng).tolist()
        shuffled = [indices[i] for i in perm]
        n_test = max(1, int(round(len(shuffled) * test_ratio))) if len(shuffled) > 1 else 0
        test_indices.extend(shuffled[:n_test])
    return test_indices


# ──────────────────────────────────────────────────────────────────────────────
# Core Evaluation Logic
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model_type: str,
    data_dir: str,
    seed: int = 42,
    batch_size: int = 32,
    results_dir: str = "results",
) -> dict:
    """
    Evaluate a single trained model and return a metrics dict.

    Returns:
        dict with keys: model_type, accuracy, per_class_f1, macro_f1,
                        confusion_matrix, samples_evaluated, latency_ms
    """
    device = torch.device(DEVICE)
    print(f"\n📋 Evaluating '{model_type}' on {device} …")

    # — Load model ─────────────────────────────────────────────────────────────
    model = create_backbone(model_type, num_classes=NUM_CLASSES, pretrained=False)
    weights = resolve_weights_path(model_type)
    if weights:
        model = load_checkpoint(model, weights)
        print(f"   ✅ Loaded weights: {weights}")
    else:
        print("   ⚠️  No fine-tuned weights found — using ImageNet-head (random classifier).")

    model = model.to(device)
    model.eval()

    # — Dataset ────────────────────────────────────────────────────────────────
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Dataset directory '{data_dir}' not found! "
            "Run 'python download_dataset.py' first."
        )

    transform = build_eval_transform()
    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    targets = [label for _, label in full_dataset.samples]
    test_indices = _stratified_test_split(targets, test_ratio=0.15, seed=seed)
    test_ds = Subset(full_dataset, test_indices)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"   📊 Test set: {len(test_indices)} samples")

    # — Inference ──────────────────────────────────────────────────────────────
    all_preds: list[int] = []
    all_labels: list[int] = []
    latencies: list[float] = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            t0 = time.perf_counter()
            outputs = model(images)
            latencies.append((time.perf_counter() - t0) * 1000.0 / images.size(0))
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())

    # — Metrics ────────────────────────────────────────────────────────────────
    n = len(all_labels)
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    accuracy = correct / n if n > 0 else 0.0

    # Confusion matrix
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for pred, label in zip(all_preds, all_labels):
        cm[label][pred] += 1

    # Per-class precision, recall, F1
    per_class_metrics: dict[str, dict] = {}
    f1_scores: list[float] = []
    for i, cls in enumerate(CLASS_LABELS):
        tp = int(cm[i][i])
        fp = int(cm[:, i].sum()) - tp
        fn = int(cm[i, :].sum()) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        per_class_metrics[cls] = {
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
            "support":   int(cm[i, :].sum()),
        }
        f1_scores.append(f1)

    macro_f1 = float(np.mean(f1_scores))
    mean_lat  = float(np.mean(latencies))

    print(f"\n   {'Class':<14}  Precision  Recall    F1")
    print(f"   {'─'*50}")
    for cls, m in per_class_metrics.items():
        print(f"   {cls:<14}  {m['precision']:.4f}     {m['recall']:.4f}    {m['f1']:.4f}")
    print(f"   {'─'*50}")
    print(f"   Top-1 Accuracy : {accuracy*100:.2f}%")
    print(f"   Macro F1       : {macro_f1*100:.2f}%")
    print(f"   Mean Latency   : {mean_lat:.2f} ms/sample")

    return {
        "model_type":       model_type,
        "accuracy":         round(accuracy, 4),
        "macro_f1":         round(macro_f1, 4),
        "per_class":        per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "class_labels":     CLASS_LABELS,
        "samples_evaluated":n,
        "mean_latency_ms":  round(mean_lat, 2),
        "seed":             seed,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Results Persistence
# ──────────────────────────────────────────────────────────────────────────────

def save_results(metrics: dict, results_dir: str) -> None:
    """Save evaluation metrics to JSON and a plain-text summary."""
    os.makedirs(results_dir, exist_ok=True)
    model_type = metrics["model_type"]

    json_path = os.path.join(results_dir, f"{model_type}_eval.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n   💾 JSON metrics saved → {json_path}")

    # Human-readable summary
    txt_path = os.path.join(results_dir, f"{model_type}_eval_summary.txt")
    lines = [
        f"OpticBin Evaluation Report — {model_type}",
        "=" * 50,
        f"Top-1 Accuracy : {metrics['accuracy']*100:.2f}%",
        f"Macro F1       : {metrics['macro_f1']*100:.2f}%",
        f"Samples        : {metrics['samples_evaluated']}",
        f"Mean Latency   : {metrics['mean_latency_ms']} ms/sample",
        f"Seed           : {metrics['seed']}",
        "",
        "Per-Class Results:",
        f"  {'Class':<14}  Precision  Recall    F1      Support",
        "  " + "─" * 54,
    ]
    for cls, m in metrics["per_class"].items():
        lines.append(
            f"  {cls:<14}  {m['precision']:.4f}     {m['recall']:.4f}    "
            f"{m['f1']:.4f}  {m['support']}"
        )
    lines.append("")
    lines.append("Confusion Matrix (rows=actual, cols=predicted):")
    lines.append("  " + "  ".join(f"{c[:4]:>4}" for c in metrics["class_labels"]))
    for i, row in enumerate(metrics["confusion_matrix"]):
        lines.append(f"  {metrics['class_labels'][i][:4]:>4} " + "  ".join(f"{v:>4}" for v in row))

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"   📄 Text summary saved  → {txt_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # Load YAML defaults
    cfg = load_yaml_config()
    default_seed = cfg.get("training", {}).get("seed", 42)
    default_data = cfg.get("paths", {}).get("dataset_dir", "dataset")
    default_results = cfg.get("paths", {}).get("results_dir", "results")

    parser = argparse.ArgumentParser(description="OpticBin Model Evaluator")
    parser.add_argument(
        "--model",
        type=str,
        default="efficientnetv2_s",
        choices=["efficientnetv2_s", "mobilevit_xs", "all"],
        help="Backbone to evaluate",
    )
    parser.add_argument("--data_dir", type=str, default=default_data, help="Dataset directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--seed", type=int, default=default_seed, help="Random seed")
    parser.add_argument("--results_dir", type=str, default=default_results, help="Output directory for metrics")
    args = parser.parse_args()

    set_global_seed(args.seed)

    models_to_eval = (
        ["efficientnetv2_s", "mobilevit_xs"] if args.model == "all" else [args.model]
    )

    all_results = []
    for model_name in models_to_eval:
        print(f"\n{'='*48}")
        print(f"  Model: {model_name}")
        print(f"{'='*48}")
        metrics = evaluate_model(
            model_type=model_name,
            data_dir=args.data_dir,
            seed=args.seed,
            batch_size=args.batch_size,
            results_dir=args.results_dir,
        )
        save_results(metrics, results_dir=args.results_dir)
        all_results.append(metrics)

    # Save combined summary if evaluating both models
    if len(all_results) > 1:
        combined_path = os.path.join(args.results_dir, "all_models_eval.json")
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n📦 Combined results saved → {combined_path}")

    print("\n✅ Evaluation complete.")


if __name__ == "__main__":
    main()
