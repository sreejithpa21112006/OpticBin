"""
OpticBin — Model Evaluation Script
===================================
Evaluates trained models on the validation set and saves metrics to results/.

Usage:
    python evaluate.py --model efficientnetv2_s
    python evaluate.py --model mobilevit_xs
    python evaluate.py --model all
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

from config.settings import (
    CLASS_LABELS,
    DATASET_DIR,
    DEFAULT_SEED,
    DEFAULT_VAL_RATIO,
    DEVICE,
    NUM_CLASSES,
    RESULTS_DIR,
    WEIGHTS_DIR,
)
from src.model_factory import create_backbone, load_checkpoint, resolve_weights_path
from src.preprocessor import build_eval_transform
from src.trainer import _stratified_split


def evaluate_model(
    model_type: str,
    data_dir: str = DATASET_DIR,
    batch_size: int = 32,
    seed: int = DEFAULT_SEED,
) -> dict:
    """
    Evaluate a trained model and return comprehensive metrics.
    """
    device = torch.device(DEVICE)
    
    weights_path = resolve_weights_path(model_type, WEIGHTS_DIR)
    if weights_path is None:
        print(f"[ERROR] No trained weights found for '{model_type}' in '{WEIGHTS_DIR}'")
        print("        Run 'python train.py' first to train the model.")
        return {}

    print(f"🔍 Evaluating model: {model_type}")
    print(f"📂 Dataset: {data_dir}")
    print(f"💻 Device: {device}")
    print(f"📦 Weights: {weights_path}")
    print()

    model = create_backbone(model_type, num_classes=NUM_CLASSES, pretrained=False)
    model = load_checkpoint(model, weights_path, map_location=device)
    model = model.to(device).eval()

    val_transform = build_eval_transform()
    dataset = datasets.ImageFolder(root=data_dir, transform=val_transform)
    
    targets = [label for _, label in dataset.samples]
    _, val_indices = _stratified_split(targets, val_ratio=DEFAULT_VAL_RATIO, seed=seed)
    
    val_dataset = Subset(dataset, val_indices)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"📊 Validation samples: {len(val_indices)}")
    print()

    all_preds = []
    all_labels = []
    all_probs = []
    latencies = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            
            start_time = time.perf_counter()
            outputs = model(images)
            latency_ms = (time.perf_counter() - start_time) * 1000
            latencies.append(latency_ms / len(images))
            
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()
            
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
            all_probs.extend(probs.tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    conf_matrix = confusion_matrix(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds, target_names=CLASS_LABELS, output_dict=True)

    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    p95_latency = np.percentile(latencies, 95)

    metrics = {
        "model_type": model_type,
        "weights_path": str(weights_path),
        "evaluation_timestamp": datetime.now().isoformat(),
        "device": str(device),
        "num_samples": len(val_indices),
        "seed": seed,
        "overall_metrics": {
            "accuracy": round(accuracy, 4),
            "precision_weighted": round(precision, 4),
            "recall_weighted": round(recall, 4),
            "f1_weighted": round(f1, 4),
        },
        "latency_ms": {
            "mean": round(avg_latency, 2),
            "std": round(std_latency, 2),
            "p95": round(p95_latency, 2),
        },
        "per_class_metrics": {
            label: {
                "precision": round(class_report[label]["precision"], 4),
                "recall": round(class_report[label]["recall"], 4),
                "f1-score": round(class_report[label]["f1-score"], 4),
                "support": int(class_report[label]["support"]),
            }
            for label in CLASS_LABELS
        },
        "confusion_matrix": conf_matrix.tolist(),
    }

    print("=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Model:           {model_type}")
    print(f"Accuracy:        {accuracy * 100:.2f}%")
    print(f"Precision:       {precision * 100:.2f}%")
    print(f"Recall:          {recall * 100:.2f}%")
    print(f"F1 Score:        {f1 * 100:.2f}%")
    print(f"Avg Latency:     {avg_latency:.2f} ms")
    print(f"P95 Latency:     {p95_latency:.2f} ms")
    print("=" * 50)
    print()
    print("Per-Class Performance:")
    print("-" * 50)
    for label in CLASS_LABELS:
        cm = class_report[label]
        print(f"  {label:12s}: P={cm['precision']:.3f} R={cm['recall']:.3f} F1={cm['f1-score']:.3f} (n={int(cm['support'])})")
    print()

    return metrics


def save_results(metrics: dict, results_dir: str = RESULTS_DIR) -> str:
    """Save evaluation metrics to JSON file."""
    os.makedirs(results_dir, exist_ok=True)
    
    model_type = metrics.get("model_type", "unknown")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"eval_{model_type}_{timestamp}.json"
    filepath = os.path.join(results_dir, filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"📄 Results saved to: {filepath}")
    return filepath


def main() -> None:
    parser = argparse.ArgumentParser(description="OpticBin Model Evaluation CLI")
    parser.add_argument(
        "--model",
        type=str,
        default="efficientnetv2_s",
        choices=["efficientnetv2_s", "mobilevit_xs", "all"],
        help="Model to evaluate",
    )
    parser.add_argument("--data_dir", type=str, default=DATASET_DIR, help="Dataset directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to file")
    
    args = parser.parse_args()
    
    models_to_eval = ["efficientnetv2_s", "mobilevit_xs"] if args.model == "all" else [args.model]
    
    for model_name in models_to_eval:
        print("\n" + "=" * 60)
        print(f"  EVALUATING: {model_name}")
        print("=" * 60 + "\n")
        
        metrics = evaluate_model(
            model_type=model_name,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        
        if metrics and not args.no_save:
            save_results(metrics)


if __name__ == "__main__":
    main()
