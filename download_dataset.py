"""
OpticBin -- Dataset Downloader & Manager (Extended 8-Class)
============================================================
Downloads, extracts, or imports waste datasets into dataset/

Classes (8):
    cardboard, glass, metal, paper, plastic  <- original 5 (TrashNet)
    e_waste, organic, trash                  <- 3 new classes

Usage:
    python download_dataset.py                        # Downloads all 8 classes
    python download_dataset.py --info                 # Displays dataset stats
    python download_dataset.py --dataset synthetic    # Generates synthetic dataset
    python download_dataset.py --dataset multi        # TrashNet + synthetic samples
    python download_dataset.py --augment 2            # 2x offline augmentation
    python download_dataset.py --import-zip path.zip  # Imports custom ZIP
    python download_dataset.py --import-folder dir/   # Imports from folder
"""

import argparse
import math
import os
import random
import shutil
import sys
import urllib.request
import zipfile

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

DATASET_URL = "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip"
DEFAULT_DEST_DIR = "dataset"
ZIP_PATH = "trashnet.zip"

# Original 5 classes (TrashNet)
ORIGINAL_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic"]

# All 8 classes
CLASS_LABELS = ["cardboard", "glass", "metal", "paper", "plastic", "e_waste", "organic", "trash"]

# ─────────────────────────────────────────────────────────────────────────────
# Open image sources for new classes (CC0/public domain datasets)
# These use Kaggle open datasets hosted on GitHub mirrors / HuggingFace
# ─────────────────────────────────────────────────────────────────────────────
NEW_CLASS_SOURCES = {
    "e_waste": [
        # WEEE / e-waste images from open GitHub repos
        "https://github.com/AgaMiko/waste-datasets-review/raw/main/links/datasets_links.md",  # placeholder
    ],
    "organic": [],
    "trash": [],
}

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic visual profiles for new classes
# ─────────────────────────────────────────────────────────────────────────────
SYNTHETIC_PROFILES = {
    # existing 5 classes kept for completeness
    "cardboard": {
        "bg": (230, 230, 230), "shape": "rect",
        "colors": [(185, 140, 85), (170, 125, 75), (195, 150, 95)],
        "texture": "corrugation",
    },
    "glass": {
        "bg": (220, 235, 240), "shape": "ellipse",
        "colors": [(90, 195, 215), (70, 170, 200), (110, 205, 220)],
        "texture": "specular",
    },
    "metal": {
        "bg": (225, 225, 230), "shape": "rect",
        "colors": [(185, 190, 200), (170, 175, 185), (200, 205, 210)],
        "texture": "metallic",
    },
    "paper": {
        "bg": (245, 245, 242), "shape": "rect",
        "colors": [(250, 248, 242), (240, 238, 232), (255, 252, 246)],
        "texture": "lines",
    },
    "plastic": {
        "bg": (230, 230, 230), "shape": "rounded_rect",
        "colors": [(240, 190, 50), (40, 160, 220), (220, 70, 120), (50, 200, 120)],
        "texture": "highlight",
    },
    # NEW classes
    "e_waste": {
        "bg": (210, 210, 215), "shape": "rect",
        "colors": [(40, 40, 45), (55, 55, 60), (30, 35, 40), (80, 80, 85)],
        "texture": "circuit",      # dark PCB-like with circuit trace lines
    },
    "organic": {
        "bg": (210, 225, 205), "shape": "blob",
        "colors": [(110, 160, 60), (140, 100, 40), (180, 130, 50), (80, 140, 50)],
        "texture": "organic",      # irregular earthy shapes
    },
    "trash": {
        "bg": (220, 215, 210), "shape": "crumpled",
        "colors": [(180, 175, 170), (160, 155, 150), (200, 195, 190), (140, 135, 130)],
        "texture": "mixed",        # crumpled irregular shapes
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# TrashNet Download (original 5 classes)
# ─────────────────────────────────────────────────────────────────────────────

def download_trashnet(dest_dir: str = DEFAULT_DEST_DIR):
    """Downloads and extracts the 5-class TrashNet dataset."""
    os.makedirs(dest_dir, exist_ok=True)

    print("[INFO] Downloading TrashNet dataset (42 MB)...")
    try:
        urllib.request.urlretrieve(DATASET_URL, ZIP_PATH)
        print("[OK] Download complete.")

        print("[INFO] Extracting dataset...")
        try:
            with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
                zip_ref.extractall("dataset_temp")

            extracted_root = os.path.join("dataset_temp", "dataset-resized")
            search_dir = extracted_root if os.path.exists(extracted_root) else "dataset_temp"

            for folder in os.listdir(search_dir):
                src = os.path.join(search_dir, folder)
                dst = os.path.join(dest_dir, folder)
                if os.path.isdir(src):
                    os.makedirs(dst, exist_ok=True)
                    for file_name in os.listdir(src):
                        s_file = os.path.join(src, file_name)
                        d_file = os.path.join(dst, file_name)
                        shutil.copy2(s_file, d_file)
        finally:
            if os.path.exists("dataset_temp"):
                shutil.rmtree("dataset_temp", ignore_errors=True)
            if os.path.exists(ZIP_PATH):
                os.remove(ZIP_PATH)

        # Remove trash folder from TrashNet (we use our own richer version)
        trash_dir = os.path.join(dest_dir, "trash")
        if os.path.exists(trash_dir):
            shutil.rmtree(trash_dir)

        print(f"[OK] TrashNet dataset prepared in '{dest_dir}/' (5 classes)")
        print_dataset_info(dest_dir)
    except Exception as e:
        print(f"[ERROR] Could not auto-download: {e}")
        print(f"[INFO] Manually place images into subfolders inside '{dest_dir}/'")


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic Dataset Generator (all 8 classes)
# ─────────────────────────────────────────────────────────────────────────────

def _draw_class(draw, img, cls_name: str, size: int = 224):
    """Draw class-specific synthetic pattern onto PIL image."""
    from PIL import ImageDraw, ImageFilter
    profile = SYNTHETIC_PROFILES.get(cls_name, SYNTHETIC_PROFILES["trash"])
    color = random.choice(profile["colors"])
    cx, cy = size // 2 + random.randint(-20, 20), size // 2 + random.randint(-20, 20)

    texture = profile["texture"]

    if texture == "corrugation":
        # Cardboard: brown rectangle with horizontal lines
        w, h = random.randint(110, 160), random.randint(100, 150)
        draw.rectangle([cx-w//2, cy-h//2, cx+w//2, cy+h//2], fill=color, outline=(120,90,60), width=3)
        for y in range(cy-h//2+10, cy+h//2-10, 12):
            draw.line([(cx-w//2+5, y), (cx+w//2-5, y)], fill=(color[0]-25, color[1]-20, color[2]-15), width=2)

    elif texture == "specular":
        # Glass: ellipse with specular arc
        rx, ry = random.randint(40, 60), random.randint(65, 85)
        draw.ellipse([cx-rx, cy-ry, cx+rx, cy+ry], fill=color, outline=(50,100,140), width=3)
        draw.arc([cx-rx+8, cy-ry+8, cx+rx-8, cy+ry-8], start=200, end=280, fill=(255,255,255), width=4)

    elif texture == "metallic":
        # Metal can: rectangle with highlight stripe
        w, h = random.randint(70, 90), random.randint(100, 130)
        draw.rectangle([cx-w//2, cy-h//2, cx+w//2, cy+h//2], fill=color, outline=(100,105,115), width=3)
        draw.rectangle([cx-10, cy-h//2+5, cx+10, cy+h//2-5], fill=(240,245,250))

    elif texture == "lines":
        # Paper: white rectangle with text lines
        margin = random.randint(25, 45)
        draw.rectangle([margin, margin, size-margin, size-margin], fill=color, outline=(180,180,180), width=2)
        for ly in range(margin+20, size-margin-20, 15):
            draw.line([(margin+15, ly), (size-margin-20, ly)], fill=(160,160,170), width=2)

    elif texture == "highlight":
        # Plastic bottle: rounded rect with cap
        w, h = random.randint(75, 95), random.randint(85, 115)
        draw.rounded_rectangle([cx-w//2, cy-h//2, cx+w//2, cy+h//2], radius=15, fill=color, outline=(40,40,40), width=3)
        draw.rectangle([cx-15, cy-h//2-12, cx+15, cy-h//2], fill=(220,220,220), outline=(50,50,50), width=2)

    elif texture == "circuit":
        # E-waste: dark PCB-like rectangle with green circuit traces
        w, h = random.randint(100, 150), random.randint(80, 130)
        draw.rectangle([cx-w//2, cy-h//2, cx+w//2, cy+h//2], fill=color, outline=(0,180,0), width=2)
        # Circuit trace lines
        trace_color = (0, random.randint(160, 220), 0)
        for _ in range(random.randint(4, 8)):
            x1 = random.randint(cx-w//2+10, cx+w//2-10)
            y1 = random.randint(cy-h//2+10, cy+h//2-10)
            x2 = random.randint(cx-w//2+10, cx+w//2-10)
            y2 = random.randint(cy-h//2+10, cy+h//2-10)
            draw.line([(x1, y1), (x2, y2)], fill=trace_color, width=2)
        # Component circles (chips/capacitors)
        for _ in range(random.randint(2, 5)):
            bx = random.randint(cx-w//2+15, cx+w//2-15)
            by = random.randint(cy-h//2+15, cy+h//2-15)
            r = random.randint(5, 12)
            draw.ellipse([bx-r, by-r, bx+r, by+r], fill=(80,80,80), outline=(200,200,0), width=1)
        # Simulate a cable/wire coming out
        wire_x = cx + random.randint(-20, 20)
        draw.line([(wire_x, cy+h//2), (wire_x+random.randint(-30,30), cy+h//2+40)],
                  fill=(100,100,100), width=4)

    elif texture == "organic":
        # Organic: irregular blob shapes in earthy tones
        for _ in range(random.randint(2, 4)):
            bx = cx + random.randint(-40, 40)
            by = cy + random.randint(-40, 40)
            rx = random.randint(20, 50)
            ry = random.randint(20, 45)
            shade = (max(0, color[0]+random.randint(-20,20)),
                     max(0, color[1]+random.randint(-20,20)),
                     max(0, color[2]+random.randint(-20,20)))
            draw.ellipse([bx-rx, by-ry, bx+rx, by+ry], fill=shade, outline=(80,60,20), width=2)

    elif texture == "mixed":
        # Trash: crumpled bag / irregular polygon
        points = [(cx + random.randint(-60, 60), cy + random.randint(-60, 60)) for _ in range(7)]
        draw.polygon(points, fill=color, outline=(100,95,90), width=2)
        # Random scribble marks
        for _ in range(3):
            x1, y1 = cx + random.randint(-40,40), cy + random.randint(-40,40)
            draw.line([(x1, y1), (x1+random.randint(-30,30), y1+random.randint(-30,30))],
                      fill=(90,85,80), width=2)


def generate_synthetic_dataset(dest_dir: str = DEFAULT_DEST_DIR, samples_per_class: int = 100):
    """Generates synthetic waste images for all 8 classes."""
    os.makedirs(dest_dir, exist_ok=True)
    print(f"[INFO] Generating synthetic dataset ({samples_per_class} images/class, 8 classes)...")

    try:
        from PIL import Image, ImageDraw, ImageFilter
    except ImportError:
        print("[WARN] Pillow is required for synthetic generation.")
        return

    random.seed(42)

    for cls_name in CLASS_LABELS:
        cls_dir = os.path.join(dest_dir, cls_name)
        os.makedirs(cls_dir, exist_ok=True)
        profile = SYNTHETIC_PROFILES.get(cls_name, SYNTHETIC_PROFILES["trash"])

        for i in range(1, samples_per_class + 1):
            img_path = os.path.join(cls_dir, f"syn_{cls_name}_{i:04d}.jpg")
            if os.path.exists(img_path):
                continue

            bg_color = tuple(max(0, min(255, c + random.randint(-10, 10))) for c in profile["bg"])
            img = Image.new("RGB", (224, 224), color=bg_color)
            draw = ImageDraw.Draw(img)
            _draw_class(draw, img, cls_name, size=224)

            # Slight blur for realism
            if random.random() > 0.6:
                img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))

            img.save(img_path, quality=90)

        print(f"  [OK] {cls_name:<12}: {samples_per_class} synthetic images generated")

    print(f"\n[OK] Synthetic 8-class dataset ready in '{dest_dir}/'")
    print_dataset_info(dest_dir)


# ─────────────────────────────────────────────────────────────────────────────
# New Class Folder Setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_new_class_folders(dest_dir: str = DEFAULT_DEST_DIR):
    """
    Creates e_waste/, organic/, and trash/ folders and prints instructions
    for the user to drop in real images from Kaggle/Google.
    """
    new_classes = {
        "e_waste": [
            "https://www.kaggle.com/datasets/akshat103/e-waste-image-dataset",
            "https://www.kaggle.com/datasets/pauloviniciusornelas/ewaste",
            "Search Google Images: 'broken headphones', 'old phone', 'PCB board', 'old keyboard'",
        ],
        "organic": [
            "https://www.kaggle.com/datasets/techsash/waste-classification-data",
            "https://www.kaggle.com/datasets/mostafaabla/garbage-classification (organic folder)",
            "Search Google Images: 'banana peel', 'food scraps', 'fruit waste', 'vegetable peel'",
        ],
        "trash": [
            "https://www.kaggle.com/datasets/techsash/waste-classification-data",
            "https://www.kaggle.com/datasets/mostafaabla/garbage-classification (trash folder)",
            "Search Google Images: 'tissue paper waste', 'used disposable cup', 'mixed garbage'",
        ],
    }

    print("\n" + "="*60)
    print("  Setting up new class folders (e_waste, organic, trash)")
    print("="*60)

    for cls, sources in new_classes.items():
        folder = os.path.join(dest_dir, cls)
        os.makedirs(folder, exist_ok=True)
        existing = len([f for f in os.listdir(folder)
                        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))])
        print(f"\n  [{cls}] folder: {folder}  ({existing} images currently)")
        print(f"  Recommended sources:")
        for s in sources:
            print(f"    - {s}")

    print("\n" + "="*60)
    print("  INSTRUCTIONS:")
    print("  1. Download images from the sources above")
    print("  2. Drop them into the respective folders:")
    print(f"     dataset/e_waste/   <- electronics images")
    print(f"     dataset/organic/   <- food/plant waste images")
    print(f"     dataset/trash/     <- general non-recyclable images")
    print("  3. Aim for 400-800 images per new class (match existing classes)")
    print("  4. Then run: python download_dataset.py --augment 2")
    print("  5. Then retrain: python train.py --model all --epochs 15 --seed 42")
    print("="*60)

    # Generate 100 synthetic placeholders per new class to allow immediate training
    print("\n[INFO] Generating 100 synthetic placeholder images per new class...")
    generate_synthetic_dataset(dest_dir=dest_dir, samples_per_class=100)


# ─────────────────────────────────────────────────────────────────────────────
# Import Helpers
# ─────────────────────────────────────────────────────────────────────────────

def import_custom_zip(zip_path: str, dest_dir: str = DEFAULT_DEST_DIR):
    """Extracts a custom dataset ZIP archive into dest_dir."""
    if not os.path.exists(zip_path):
        print(f"[ERROR] ZIP file not found: '{zip_path}'")
        return

    os.makedirs(dest_dir, exist_ok=True)
    print(f"[INFO] Extracting custom ZIP dataset from '{zip_path}'...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(dest_dir)
        print(f"[OK] Custom dataset extracted to '{dest_dir}/'")
        print_dataset_info(dest_dir)
    except (zipfile.BadZipFile, OSError) as e:
        print(f"[ERROR] Failed to extract ZIP: {e}")


def import_custom_folder(folder_path: str, dest_dir: str = DEFAULT_DEST_DIR):
    """Copies subfolders from a custom directory into dest_dir."""
    if not os.path.exists(folder_path):
        print(f"[ERROR] Source folder not found: '{folder_path}'")
        return

    os.makedirs(dest_dir, exist_ok=True)
    print(f"[INFO] Importing dataset from folder '{folder_path}'...")
    try:
        for item in os.listdir(folder_path):
            src = os.path.join(folder_path, item)
            dst = os.path.join(dest_dir, item)
            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            elif os.path.isfile(src):
                shutil.copy2(src, dst)
        print(f"[OK] Dataset imported to '{dest_dir}/'")
        print_dataset_info(dest_dir)
    except OSError as e:
        print(f"[ERROR] Failed to import folder: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation
# ─────────────────────────────────────────────────────────────────────────────

def augment_dataset(dest_dir: str = DEFAULT_DEST_DIR, multiplier: int = 2):
    """Expands dataset by generating offline augmented images for all class folders."""
    if not os.path.exists(dest_dir):
        print(f"[ERROR] Dataset directory '{dest_dir}' does not exist.")
        return

    try:
        from PIL import Image, ImageEnhance, ImageOps
    except ImportError:
        print("[ERROR] PIL is required for offline augmentation.")
        return

    print(f"[INFO] Augmenting dataset in '{dest_dir}' with {multiplier}x factor...")
    subdirs = [
        d for d in os.listdir(dest_dir)
        if os.path.isdir(os.path.join(dest_dir, d)) and not d.startswith(".")
    ]

    total_created = 0
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    for folder in sorted(subdirs):
        folder_path = os.path.join(dest_dir, folder)
        images = [f for f in os.listdir(folder_path)
                  if os.path.splitext(f)[1].lower() in image_extensions
                  and "_aug" not in f]  # skip already augmented
        original_count = len(images)
        created_count = 0

        for img_name in images:
            img_path = os.path.join(folder_path, img_name)
            try:
                from PIL import Image, ImageEnhance
                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    base_name, ext = os.path.splitext(img_name)

                    for m in range(1, multiplier):
                        aug_name = f"{base_name}_aug{m}{ext}"
                        aug_path = os.path.join(folder_path, aug_name)
                        if os.path.exists(aug_path):
                            continue

                        aug_img = img.copy()
                        if random.random() > 0.5:
                            aug_img = aug_img.transpose(Image.FLIP_LEFT_RIGHT)
                        angle = random.uniform(-25, 25)
                        aug_img = aug_img.rotate(angle, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
                        aug_img = ImageEnhance.Brightness(aug_img).enhance(random.uniform(0.8, 1.2))
                        aug_img = ImageEnhance.Contrast(aug_img).enhance(random.uniform(0.8, 1.2))
                        aug_img.save(aug_path, quality=90)
                        created_count += 1
            except Exception:
                continue

        total_created += created_count
        print(f"  - {folder:<15}: {original_count} original -> +{created_count} augmented")

    print(f"\n[OK] Augmentation complete! Added {total_created} new images.")
    print_dataset_info(dest_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Info
# ─────────────────────────────────────────────────────────────────────────────

def print_dataset_info(dest_dir: str = DEFAULT_DEST_DIR):
    """Prints breakdown of images per class."""
    if not os.path.exists(dest_dir):
        print(f"[WARN] Dataset directory '{dest_dir}' does not exist.")
        return

    subdirs = [
        d for d in os.listdir(dest_dir)
        if os.path.isdir(os.path.join(dest_dir, d)) and not d.startswith(".")
    ]
    if not subdirs:
        print(f"[WARN] No class folders found inside '{dest_dir}'.")
        return

    print(f"\n[INFO] Dataset Summary ('{dest_dir}/'):")
    print("=" * 45)
    total_images = 0
    for folder in sorted(subdirs):
        folder_path = os.path.join(dest_dir, folder)
        image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff")
        count = sum(
            1 for f in os.listdir(folder_path)
            if os.path.splitext(f)[1].lower() in image_extensions
        )
        total_images += count
        tag = " [NEW]" if folder in ["e_waste", "organic", "trash"] else ""
        print(f"  - {folder:<15}: {count:>5} images{tag}")
    print("=" * 45)
    print(f"  Total: {total_images} images across {len(subdirs)} classes.\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

# CLI
# ───────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OpticBin Dataset Downloader & Manager (8-class)")
    parser.add_argument(
        "--dataset",
        type=str,
        default="trashnet",
        choices=["trashnet", "synthetic", "multi", "combined"],
        help="Dataset source to prepare (default: trashnet)",
    )
    parser.add_argument("--dest", type=str, default=DEFAULT_DEST_DIR, help="Destination directory")
    parser.add_argument("--samples-per-class", type=int, default=100, help="Synthetic samples per class")
    parser.add_argument("--import-zip", type=str, default=None, help="Path to custom ZIP file to import")
    parser.add_argument("--import-folder", type=str, default=None, help="Path to custom folder to import")
    parser.add_argument("--augment", type=int, default=0, help="Offline augmentation multiplier (e.g. 2 or 3)")
    parser.add_argument("--info", action="store_true", help="Display dataset class statistics")
    parser.add_argument("--setup-new-classes", action="store_true",
                        help="Create e_waste/organic/trash folders + synthetic placeholders + show sources")
    parser.add_argument("--download-kaggle", action="store_true",
                        help="Auto-download e_waste/organic/trash from Kaggle using kagglehub")

    args = parser.parse_args()

    if args.info:
        print_dataset_info(args.dest)
    elif args.download_kaggle:
        download_kaggle_new_classes(args.dest)
    elif args.setup_new_classes:
        setup_new_class_folders(args.dest)
    elif args.augment > 1:
        augment_dataset(dest_dir=args.dest, multiplier=args.augment)
    elif args.import_zip:
        import_custom_zip(args.import_zip, dest_dir=args.dest)
    elif args.import_folder:
        import_custom_folder(args.import_folder, dest_dir=args.dest)
    elif args.dataset in ["multi", "combined"]:
        download_trashnet(dest_dir=args.dest)
        generate_synthetic_dataset(dest_dir=args.dest, samples_per_class=args.samples_per_class)
        setup_new_class_folders(args.dest)
    elif args.dataset == "synthetic":
        generate_synthetic_dataset(dest_dir=args.dest, samples_per_class=args.samples_per_class)
    else:
        download_trashnet(dest_dir=args.dest)
        setup_new_class_folders(args.dest)


if __name__ == "__main__":
    main()
