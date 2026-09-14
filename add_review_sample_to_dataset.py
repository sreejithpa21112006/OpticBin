"""
Copies Gemini-verified edge cases from review_queue/ into dataset_roboflow/train/
with proper normalized YOLO bounding box annotations so the model learns from past misclassifications.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from PIL import Image

ROBOFLOW_CLASSES = ["BIODEGRADABLE", "CARDBOARD", "GLASS", "METAL", "PAPER", "PLASTIC"]
CLASS_MAP = {c.lower(): i for i, c in enumerate(ROBOFLOW_CLASSES)}

REVIEW_DIR = Path("review_queue")
TRAIN_IMG_DIR = Path("dataset_roboflow/train/images")
TRAIN_LBL_DIR = Path("dataset_roboflow/train/labels")


def add_verified_samples():
    if not REVIEW_DIR.exists():
        print("[INFO] No review queue found.")
        return

    added = 0
    for folder in REVIEW_DIR.iterdir():
        if not folder.is_dir():
            continue
        meta_file = folder / "metadata.json"
        img_file = folder / "image.jpg"
        if not meta_file.exists() or not img_file.exists():
            continue

        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)

            verified_label = (meta.get("verified_label") or meta.get("gemini_label") or "").lower().strip()
            gemini_conf = float(meta.get("gemini_confidence", 0.0))

            if verified_label not in CLASS_MAP or gemini_conf < 0.70:
                continue

            class_id = CLASS_MAP[verified_label]

            # Get image dimensions
            with Image.open(img_file) as im:
                w, h = im.size

            # If metadata has primary box, use it; otherwise center object box (0.7 scale)
            box = meta.get("primary_box")
            if box and len(box) == 4:
                x1, y1, x2, y2 = box
                xc = (x1 + x2) / (2.0 * w)
                yc = (y1 + y2) / (2.0 * h)
                bw = (x2 - x1) / float(w)
                bh = (y2 - y1) / float(h)
            else:
                xc, yc, bw, bh = 0.50, 0.50, 0.70, 0.85

            stem = f"review_{folder.name}"
            dest_img = TRAIN_IMG_DIR / f"{stem}.jpg"
            dest_lbl = TRAIN_LBL_DIR / f"{stem}.txt"

            shutil.copy2(img_file, dest_img)
            with open(dest_lbl, "w", encoding="utf-8") as lf:
                lf.write(f"{class_id} {xc:.4f} {yc:.4f} {bw:.4f} {bh:.4f}\n")

            print(f"[OK] Added {stem} as class {class_id} ({verified_label.upper()}) to training set.")
            added += 1

        except Exception as err:
            print(f"[WARN] Failed to process {folder.name}: {err}")

    print(f"[DONE] Incorporated {added} verified edge cases into dataset_roboflow/train/")


if __name__ == "__main__":
    add_verified_samples()
