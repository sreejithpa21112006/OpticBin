"""
OpticBin — Convert Classification Dataset to YOLO Bounding Box Format
======================================================================
Converts folder-structured classification datasets (dataset/cardboard, dataset/glass,
dataset/e_waste, etc.) into YOLOv8 object detection format with normalized bounding boxes.
Merges converted samples into dataset_roboflow/ for retraining.

Usage:
    python convert_dataset_to_yolo.py --dry-run
    python convert_dataset_to_yolo.py --limit-per-class 500
    python convert_dataset_to_yolo.py --all
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# 7-Class Vocabulary aligning with dataset_roboflow + e_waste
YOLO_CLASSES = [
    "BIODEGRADABLE",  # 0: organic, food, compost
    "CARDBOARD",      # 1: cardboard boxes
    "GLASS",          # 2: glass bottles, jars
    "METAL",          # 3: cans, foil, tins
    "PAPER",          # 4: sheets, newspaper, magazines
    "PLASTIC",        # 5: plastic bottles, containers
    "E-WASTE",        # 6: batteries, electronics, pcbs, cables
]

# Mapping from dataset/ subfolder names to YOLO class indices
FOLDER_TO_CLASS = {
    "organic": 0,
    "biodegradable": 0,
    "cardboard": 1,
    "glass": 2,
    "metal": 3,
    "paper": 4,
    "plastic": 5,
    "trash": 5,        # general synthetic trash maps into non-bio / plastic
    "e_waste": 6,
    "ewaste": 6,
}

DATASET_SRC = Path("dataset")
ROBOFLOW_DIR = Path("dataset_roboflow")
DATA_YAML = ROBOFLOW_DIR / "data.yaml"


def detect_bounding_box(img_path: Path) -> tuple[float, float, float, float]:
    """
    Computes normalized YOLO bounding box (x_center, y_center, width, height).
    Uses Otsu thresholding and contour detection for clean foreground objects,
    falling back to centered bounding box if contour detection is uncertain.
    """
    try:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            return 0.50, 0.50, 0.75, 0.80

        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Apply slight blur to reduce compression noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Invert Otsu thresholding (objects against light background)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find external contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find largest contour by area
            largest_c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_c)
            img_area = float(w * h)

            # Accept contour if it covers a sensible portion of the frame (between 8% and 92%)
            if 0.08 * img_area <= area <= 0.92 * img_area:
                bx, by, bw, bh = cv2.boundingRect(largest_c)
                # Add 5% padding around detected contour
                pad_x = int(bw * 0.05)
                pad_y = int(bh * 0.05)
                bx1 = max(0, bx - pad_x)
                by1 = max(0, by - pad_y)
                bx2 = min(w, bx + bw + pad_x)
                by2 = min(h, by + bh + pad_y)

                xc = ((bx1 + bx2) / 2.0) / float(w)
                yc = ((by1 + by2) / 2.0) / float(h)
                nw = (bx2 - bx1) / float(w)
                nh = (by2 - by1) / float(h)
                return round(xc, 4), round(yc, 4), round(nw, 4), round(nh, 4)

    except Exception:
        pass

    # Safe default: centered object box
    return 0.50, 0.50, 0.75, 0.80


def update_data_yaml(yaml_path: Path) -> None:
    """Ensure data.yaml declares all 7 classes including E-WASTE."""
    if not yaml_path.exists():
        print(f"[WARN] {yaml_path} does not exist.")
        return

    content = f"""names:
- BIODEGRADABLE
- CARDBOARD
- GLASS
- METAL
- PAPER
- PLASTIC
- E-WASTE
nc: {len(YOLO_CLASSES)}
roboflow:
  license: CC BY 4.0
  project: garbage-classification-3
  url: https://universe.roboflow.com/material-identification/garbage-classification-3/dataset/2
  version: 2
  workspace: material-identification
test: ../test/images
train: ../train/images
val: ../valid/images
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[OK] Updated {yaml_path} with 7 classes (including E-WASTE).")


def convert_and_merge(
    src_dir: Path = DATASET_SRC,
    target_dir: Path = ROBOFLOW_DIR,
    limit_per_class: int | None = None,
    val_split: float = 0.15,
    dry_run: bool = False,
) -> None:
    """
    Main ingestion routine: converts classification dataset into YOLO detection format.
    """
    if not src_dir.exists():
        print(f"[ERROR] Source dataset directory '{src_dir.resolve()}' not found.")
        return

    print("=" * 65)
    print(" OpticBin - Classification Dataset to YOLO Bbox Converter")
    print("=" * 65)
    print(f"Source Directory   : {src_dir.resolve()}")
    print(f"Target Directory   : {target_dir.resolve()}")
    print(f"Limit per category : {'All images' if limit_per_class is None else limit_per_class}")
    print(f"Validation split   : {val_split * 100:.0f}%")
    print(f"Dry Run Mode       : {dry_run}")
    print("=" * 65 + "\n")

    train_img_dir = target_dir / "train" / "images"
    train_lbl_dir = target_dir / "train" / "labels"
    val_img_dir = target_dir / "valid" / "images"
    val_lbl_dir = target_dir / "valid" / "labels"

    if not dry_run:
        train_img_dir.mkdir(parents=True, exist_ok=True)
        train_lbl_dir.mkdir(parents=True, exist_ok=True)
        val_img_dir.mkdir(parents=True, exist_ok=True)
        val_lbl_dir.mkdir(parents=True, exist_ok=True)

    total_processed = 0
    class_counts: dict[str, int] = {}

    subfolders = [f for f in sorted(src_dir.iterdir()) if f.is_dir() and f.name.lower() in FOLDER_TO_CLASS]

    for folder in subfolders:
        cat_name = folder.name.lower()
        class_id = FOLDER_TO_CLASS[cat_name]
        class_name = YOLO_CLASSES[class_id]

        img_files = (
            list(folder.glob("*.jpg"))
            + list(folder.glob("*.jpeg"))
            + list(folder.glob("*.png"))
        )

        if not img_files:
            continue

        # Shuffle deterministically
        random.seed(42)
        random.shuffle(img_files)

        if limit_per_class is not None:
            img_files = img_files[:limit_per_class]

        cat_processed = 0
        for img_path in img_files:
            # Decide train vs val split
            is_val = (random.random() < val_split)
            dest_img_dir = val_img_dir if is_val else train_img_dir
            dest_lbl_dir = val_lbl_dir if is_val else train_lbl_dir

            stem = f"ds_{cat_name}_{img_path.stem}"
            dest_img = dest_img_dir / f"{stem}.jpg"
            dest_lbl = dest_lbl_dir / f"{stem}.txt"

            # Compute bounding box
            xc, yc, bw, bh = detect_bounding_box(img_path)

            if not dry_run:
                try:
                    shutil.copy2(img_path, dest_img)
                    with open(dest_lbl, "w", encoding="utf-8") as lf:
                        lf.write(f"{class_id} {xc:.4f} {yc:.4f} {bw:.4f} {bh:.4f}\n")
                except Exception as err:
                    print(f"  [WARN] Failed to copy {img_path.name}: {err}")
                    continue

            cat_processed += 1

        class_counts[class_name] = class_counts.get(class_name, 0) + cat_processed
        total_processed += cat_processed
        print(f"  [OK] Processed {cat_processed:>4} images from '{folder.name}' -> Class {class_id} ({class_name})")

    if not dry_run:
        update_data_yaml(DATA_YAML)

    print("\n" + "=" * 65)
    print(" Ingestion Summary")
    print("=" * 65)
    for c_name, count in sorted(class_counts.items()):
        print(f"  - {c_name:<16} : {count:>5} images added")
    print(f"\n  Total samples converted: {total_processed}")
    print("=" * 65)


def main() -> None:
    parser = argparse.ArgumentParser(description="OpticBin Classification to YOLO Converter")
    parser.add_argument("--src", type=str, default=str(DATASET_SRC), help="Source dataset directory")
    parser.add_argument("--dest", type=str, default=str(ROBOFLOW_DIR), help="Destination Roboflow directory")
    parser.add_argument("--limit-per-class", type=int, default=None, help="Limit number of images per category")
    parser.add_argument("--val-split", type=float, default=0.15, help="Fraction to assign to validation split")
    parser.add_argument("--dry-run", action="store_true", help="Print summary without copying files")
    args = parser.parse_args()

    convert_and_merge(
        src_dir=Path(args.src),
        target_dir=Path(args.dest),
        limit_per_class=args.limit_per_class,
        val_split=args.val_split,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
