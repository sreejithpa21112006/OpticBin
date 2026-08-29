"""
OpticBin -- Dataset Downloader & Manager
========================================
Downloads, extracts, or imports waste datasets into dataset/

Usage:
    python download_dataset.py                       # Downloads TrashNet (default 5-class)
    python download_dataset.py --info                # Displays dataset stats & class counts
    python download_dataset.py --dataset synthetic   # Generates procedural synthetic dataset
    python download_dataset.py --dataset multi       # Downloads TrashNet + generates synthetic samples
    python download_dataset.py --import-zip path.zip # Imports dataset from ZIP archive
    python download_dataset.py --import-folder dir/  # Imports dataset from directory
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
CLASS_LABELS = ["cardboard", "glass", "metal", "paper", "plastic"]


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

        # Remove trash folder if present
        trash_dir = os.path.join(dest_dir, "trash")
        if os.path.exists(trash_dir):
            shutil.rmtree(trash_dir)

        print(f"[OK] TrashNet dataset prepared in '{dest_dir}/' (5 classes)")
        print_dataset_info(dest_dir)
    except (urllib.error.URLError, zipfile.BadZipFile, OSError) as e:
        print(f"[ERROR] Could not auto-download: {e}")
        print(f"[INFO] You can manually place images into subfolders inside '{dest_dir}/':")
        print("   dataset/glass, dataset/paper, dataset/cardboard, dataset/plastic, dataset/metal")


def generate_synthetic_dataset(dest_dir: str = DEFAULT_DEST_DIR, samples_per_class: int = 50):
    """Generates procedural synthetic waste images with realistic textures, highlights, and geometries."""
    os.makedirs(dest_dir, exist_ok=True)
    print(f"[INFO] Generating high-variability synthetic dataset ({samples_per_class} images per class)...")

    try:
        from PIL import Image, ImageDraw, ImageFilter
    except ImportError:
        print("[WARN] Pillow is required for synthetic generation.")
        return

    random.seed(42)

    for cls_name in CLASS_LABELS:
        cls_dir = os.path.join(dest_dir, cls_name)
        os.makedirs(cls_dir, exist_ok=True)

        for i in range(1, samples_per_class + 1):
            img_path = os.path.join(cls_dir, f"syn_{cls_name}_{i:04d}.jpg")
            if os.path.exists(img_path):
                continue

            # Base background with subtle color variations
            bg_color = (random.randint(220, 245), random.randint(220, 245), random.randint(220, 245))
            img = Image.new("RGB", (224, 224), color=bg_color)
            draw = ImageDraw.Draw(img)

            # Class-specific procedural styling
            if cls_name == "cardboard":
                # Warm brown tone with corrugation texture lines
                base_c = (random.randint(170, 200), random.randint(130, 150), random.randint(80, 100))
                bbox = [random.randint(20, 50), random.randint(20, 50), random.randint(170, 200), random.randint(170, 200)]
                draw.rectangle(bbox, fill=base_c, outline=(120, 90, 60), width=3)
                for step in range(bbox[1] + 10, bbox[3] - 10, 12):
                    draw.line([(bbox[0] + 5, step), (bbox[2] - 5, step)], fill=(base_c[0] - 25, base_c[1] - 20, base_c[2] - 15), width=2)

            elif cls_name == "glass":
                # Translucent blue/green cyan glass bottle shape with specular highlights
                glass_tint = (random.randint(80, 140), random.randint(180, 220), random.randint(200, 240))
                cx, cy = 112 + random.randint(-15, 15), 112 + random.randint(-15, 15)
                rx, ry = random.randint(35, 55), random.randint(65, 85)
                draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=glass_tint, outline=(50, 100, 140), width=3)
                # Specular reflection arc
                draw.arc([cx - rx + 8, cy - ry + 8, cx + rx - 8, cy + ry - 8], start=200, end=280, fill=(255, 255, 255), width=4)

            elif cls_name == "metal":
                # Metallic silver/gray pop can shape with metallic gradient lines
                cx, cy = 112 + random.randint(-10, 10), 112 + random.randint(-10, 10)
                w, h = random.randint(70, 90), random.randint(100, 130)
                draw.rectangle([cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2], fill=(190, 195, 205), outline=(100, 105, 115), width=3)
                # Metallic sheen highlight stripes
                draw.rectangle([cx - 10, cy - h // 2 + 5, cx + 10, cy + h // 2 - 5], fill=(240, 245, 250))

            elif cls_name == "paper":
                # White/off-white sheet geometry with fold lines and text line markings
                margin = random.randint(25, 45)
                points = [
                    (margin, margin),
                    (224 - margin - random.randint(0, 15), margin),
                    (224 - margin, 224 - margin),
                    (margin + random.randint(0, 15), 224 - margin),
                ]
                draw.polygon(points, fill=(250, 248, 242), outline=(180, 180, 180), width=2)
                # Lines representing text on paper
                for line_y in range(margin + 20, 224 - margin - 20, 15):
                    draw.line([(margin + 15, line_y), (224 - margin - 20, line_y)], fill=(160, 160, 170), width=2)

            elif cls_name == "plastic":
                # Bright yellow/orange/blue plastic container geometry
                p_colors = [(240, 190, 50), (40, 160, 220), (220, 70, 120), (50, 200, 120)]
                p_color = random.choice(p_colors)
                cx, cy = 112 + random.randint(-15, 15), 112 + random.randint(-15, 15)
                w, h = random.randint(75, 95), random.randint(85, 115)
                draw.rounded_rectangle([cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2], radius=15, fill=p_color, outline=(40, 40, 40), width=3)
                # Plastic cap handle
                draw.rectangle([cx - 15, cy - h // 2 - 12, cx + 15, cy - h // 2], fill=(220, 220, 220), outline=(50, 50, 50), width=2)

            img.save(img_path, quality=90)

    print(f"[OK] Synthetic dataset generated successfully in '{dest_dir}/'")
    print_dataset_info(dest_dir)


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
    """Copies subfolders and files from custom directory into dest_dir."""
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


def print_dataset_info(dest_dir: str = DEFAULT_DEST_DIR):
    """Prints breakdown of images per class in dataset directory."""
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
    print("=" * 40)
    total_images = 0
    for folder in sorted(subdirs):
        folder_path = os.path.join(dest_dir, folder)
        image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff")
        count = sum(
            1 for f in os.listdir(folder_path)
            if os.path.splitext(f)[1].lower() in image_extensions
        )
        total_images += count
        print(f"  - {folder:<15}: {count:>5} images")
    print("=" * 40)
    print(f"  Total: {total_images} images across {len(subdirs)} classes.\n")


def augment_dataset(dest_dir: str = DEFAULT_DEST_DIR, multiplier: int = 2):
    """
    Expands the dataset by generating realistic offline augmented images
    (rotations, color variations, flips, zoom) for each class folder.
    """
    if not os.path.exists(dest_dir):
        print(f"[ERROR] Dataset directory '{dest_dir}' does not exist.")
        return

    try:
        from PIL import Image, ImageEnhance, ImageOps
        import random
    except ImportError:
        print("[ERROR] PIL is required for offline augmentation.")
        return

    print(f"[INFO] Augmenting dataset in '{dest_dir}' with a {multiplier}x factor...")
    subdirs = [
        d for d in os.listdir(dest_dir)
        if os.path.isdir(os.path.join(dest_dir, d)) and not d.startswith(".")
    ]

    total_created = 0
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    for folder in sorted(subdirs):
        folder_path = os.path.join(dest_dir, folder)
        images = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in image_extensions]
        original_count = len(images)
        created_count = 0

        for img_name in images:
            img_path = os.path.join(folder_path, img_name)
            try:
                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    base_name, ext = os.path.splitext(img_name)

                    for m in range(1, multiplier):
                        aug_name = f"{base_name}_aug{m}{ext}"
                        aug_path = os.path.join(folder_path, aug_name)
                        if os.path.exists(aug_path):
                            continue

                        aug_img = img.copy()

                        # Random Horizontal Flip
                        if random.random() > 0.5:
                            aug_img = aug_img.transpose(Image.FLIP_LEFT_RIGHT)

                        # Random Rotation (-25 to 25 deg)
                        angle = random.uniform(-25, 25)
                        aug_img = aug_img.rotate(angle, resample=Image.BICUBIC, fillcolor=(255, 255, 255))

                        # Color / Brightness Jitter
                        brightness = random.uniform(0.8, 1.2)
                        contrast = random.uniform(0.8, 1.2)
                        aug_img = ImageEnhance.Brightness(aug_img).enhance(brightness)
                        aug_img = ImageEnhance.Contrast(aug_img).enhance(contrast)

                        aug_img.save(aug_path, quality=90)
                        created_count += 1
            except Exception as e:
                continue

        total_created += created_count
        print(f"  - {folder:<15}: {original_count} original -> +{created_count} augmented")

    print(f"\n[OK] Dataset augmentation completed! Added {total_created} new images.")
    print_dataset_info(dest_dir)


def main():
    parser = argparse.ArgumentParser(description="OpticBin Dataset Downloader & Manager")
    parser.add_argument(
        "--dataset",
        type=str,
        default="trashnet",
        choices=["trashnet", "synthetic", "multi", "combined"],
        help="Dataset source to prepare (default: trashnet)",
    )
    parser.add_argument("--dest", type=str, default=DEFAULT_DEST_DIR, help="Destination directory")
    parser.add_argument("--samples-per-class", type=int, default=50, help="Number of synthetic samples per class")
    parser.add_argument("--import-zip", type=str, default=None, help="Path to custom ZIP file to import")
    parser.add_argument("--import-folder", type=str, default=None, help="Path to custom folder to import")
    parser.add_argument("--augment", type=int, default=0, help="Offline augmentation factor per image (e.g., 2 or 3)")
    parser.add_argument("--info", action="store_true", help="Display dataset class statistics")

    args = parser.parse_args()

    if args.info:
        print_dataset_info(args.dest)
    elif args.augment > 1:
        augment_dataset(dest_dir=args.dest, multiplier=args.augment)
    elif args.import_zip:
        import_custom_zip(args.import_zip, dest_dir=args.dest)
    elif args.import_folder:
        import_custom_folder(args.import_folder, dest_dir=args.dest)
    elif args.dataset in ["multi", "combined"]:
        download_trashnet(dest_dir=args.dest)
        generate_synthetic_dataset(dest_dir=args.dest, samples_per_class=args.samples_per_class)
    elif args.dataset == "synthetic":
        generate_synthetic_dataset(dest_dir=args.dest, samples_per_class=args.samples_per_class)
    else:
        download_trashnet(dest_dir=args.dest)


if __name__ == "__main__":
    main()

