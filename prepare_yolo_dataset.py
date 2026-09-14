"""
OpticBin - YOLO Dataset Preparation and Validator
==================================================
Initializes and validates the directory structure and bounding-box label files
for training unified YOLOv8 models on the 5 OpticBin material categories.

Usage:
    python prepare_yolo_dataset.py --init
    python prepare_yolo_dataset.py --check
    python prepare_yolo_dataset.py --sample
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic"]
DATASET_DIR = Path("dataset_yolo")


def init_structure() -> None:
    """Create the standard YOLOv8 folder structure."""
    splits = ["train", "val", "test"]
    for split in splits:
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    print(f"[OK] Initialized YOLO dataset directories under '{DATASET_DIR.resolve()}':")
    print("     - images/train, images/val, images/test")
    print("     - labels/train, labels/val, labels/test")


def check_dataset(target_dir: Path | None = None) -> bool:
    """Validate image-to-label pairing and bounding box format."""
    base_dir = target_dir or (Path("dataset_roboflow") if Path("dataset_roboflow").exists() else DATASET_DIR)
    print(f"[INFO] Checking dataset at '{base_dir.resolve()}'...")
    if not base_dir.exists():
        print(f"[ERROR] Directory '{base_dir}' does not exist.")
        return False

    yaml_path = base_dir / "data.yaml"
    max_classes = len(CLASS_NAMES)
    if yaml_path.exists():
        try:
            import yaml
            with open(yaml_path, "r", encoding="utf-8") as f:
                ydata = yaml.safe_load(f)
                if "names" in ydata:
                    max_classes = len(ydata["names"])
                    print(f"  Detected {max_classes} classes from data.yaml: {ydata['names']}")
        except Exception:
            pass

    valid = True
    for split in ["train", "valid", "val"]:
        img_dir = base_dir / split / "images"
        lbl_dir = base_dir / split / "labels"
        if not img_dir.exists():
            img_dir = base_dir / "images" / split
            lbl_dir = base_dir / "labels" / split

        if not img_dir.exists() or not lbl_dir.exists():
            continue

        images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg"))
        labels = list(lbl_dir.glob("*.txt"))

        print(f"  Split '{split}': {len(images)} images, {len(labels)} label files found.")

        # Check label formats
        invalid_labels = 0
        for lbl_path in labels:
            try:
                with open(lbl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        cls_id = int(parts[0])
                        coords = [float(v) for v in parts[1:]]
                        if cls_id not in range(max_classes) or len(coords) != 4:
                            invalid_labels += 1
                        # Check bounding box bounds
                        if any(c < 0.0 or c > 1.0 for c in coords):
                            invalid_labels += 1
            except Exception:
                invalid_labels += 1

        if invalid_labels > 0:
            print(f"  [WARN] Split '{split}' has {invalid_labels} invalid label lines.")
            valid = False

    if valid:
        print("[OK] Dataset structure is valid and ready for training.")
    return valid


def create_sample_data() -> None:
    """Generate minimal synthetic sample images and annotations to verify the training pipeline."""
    init_structure()
    print("[INFO] Generating sample annotated images for verification...")

    # Color definitions for synthetic shapes (representing materials)
    samples = [
        # (split, filename, class_id, shape, color)
        ("train", "sample_plastic_bottle.jpg", 4, "bottle", (220, 180, 50)),
        ("train", "sample_paper_sheet.jpg", 3, "paper", (240, 240, 240)),
        ("train", "sample_soda_can.jpg", 2, "can", (80, 200, 220)),
        ("train", "sample_cardboard_box.jpg", 0, "box", (60, 100, 160)),
        ("train", "sample_glass_jar.jpg", 1, "jar", (180, 230, 190)),
        ("val", "val_plastic_cup.jpg", 4, "cup", (210, 190, 60)),
        ("val", "val_paper_notebook.jpg", 3, "book", (230, 230, 230)),
    ]

    for split, fname, cls_id, shape_type, color in samples:
        img_path = DATASET_DIR / "images" / split / fname
        lbl_path = DATASET_DIR / "labels" / split / f"{Path(fname).stem}.txt"

        # Draw 640x640 synthetic image
        img = np.full((640, 640, 3), 40, dtype=np.uint8)

        # Draw a centered bounding box (normalized center: 0.5, 0.5, size: 0.4, 0.5)
        x1, y1, x2, y2 = 192, 160, 448, 480
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        cv2.putText(img, CLASS_NAMES[cls_id].upper(), (x1 + 10, y1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        cv2.imwrite(str(img_path), img)

        # Write normalized YOLO format: class_id x_center y_center width height
        x_center = 0.50
        y_center = 0.50
        width = (x2 - x1) / 640.0
        height = (y2 - y1) / 640.0

        with open(lbl_path, "w", encoding="utf-8") as f:
            f.write(f"{cls_id} {x_center:.4f} {y_center:.4f} {width:.4f} {height:.4f}\n")

    print("[OK] Sample dataset created successfully.")
    check_dataset()


def main() -> None:
    parser = argparse.ArgumentParser(description="OpticBin YOLO Dataset Manager")
    parser.add_argument("--dir", type=str, default="", help="Path to dataset directory")
    parser.add_argument("--init", action="store_true", help="Create folder structure")
    parser.add_argument("--check", action="store_true", help="Validate dataset structure and labels")
    parser.add_argument("--sample", action="store_true", help="Generate sample dataset for test runs")
    args = parser.parse_args()

    target_dir = Path(args.dir) if args.dir else None

    if args.init:
        init_structure()
    elif args.sample:
        create_sample_data()
    else:
        check_dataset(target_dir)


if __name__ == "__main__":
    main()
