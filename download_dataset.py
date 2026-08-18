"""
OpticBin -- Dataset Downloader & Manager
========================================
Downloads, extracts, or imports waste datasets into dataset/

Usage:
    python download_dataset.py                       # Downloads TrashNet (default 5-class)
    python download_dataset.py --info                # Displays dataset stats & class counts
    python download_dataset.py --dataset synthetic   # Generates sample synthetic dataset
    python download_dataset.py --import-zip path.zip # Imports dataset from ZIP archive
    python download_dataset.py --import-folder dir/  # Imports dataset from directory
"""

import argparse
import os
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
                    if os.path.exists(dst):
                        shutil.rmtree(dst)
                    shutil.move(src, dst)
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


def generate_synthetic_dataset(dest_dir: str = DEFAULT_DEST_DIR, samples_per_class: int = 20):
    """Generates synthetic colored sample images for fast offline testing."""
    os.makedirs(dest_dir, exist_ok=True)
    print(f"[INFO] Generating synthetic dataset ({samples_per_class} images per class)...")

    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("[WARN] Pillow is required for synthetic generation.")
        return

    colors = {
        "cardboard": (180, 140, 100),
        "glass": (100, 180, 240),
        "metal": (190, 190, 200),
        "paper": (240, 240, 240),
        "plastic": (240, 200, 80),
    }

    for cls_name in CLASS_LABELS:
        cls_dir = os.path.join(dest_dir, cls_name)
        os.makedirs(cls_dir, exist_ok=True)
        base_color = colors.get(cls_name, (150, 150, 150))

        for i in range(1, samples_per_class + 1):
            img_path = os.path.join(cls_dir, f"sample_{i:03d}.jpg")
            if not os.path.exists(img_path):
                img = Image.new("RGB", (224, 224), color=base_color)
                draw = ImageDraw.Draw(img)
                draw.rectangle([20, 20, 204, 204], outline=(50, 50, 50), width=3)
                img.save(img_path, quality=85)

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


def main():
    parser = argparse.ArgumentParser(description="OpticBin Dataset Downloader & Manager")
    parser.add_argument(
        "--dataset",
        type=str,
        default="trashnet",
        choices=["trashnet", "synthetic"],
        help="Dataset source to prepare (default: trashnet)",
    )
    parser.add_argument("--dest", type=str, default=DEFAULT_DEST_DIR, help="Destination directory")
    parser.add_argument("--import-zip", type=str, default=None, help="Path to custom ZIP file to import")
    parser.add_argument("--import-folder", type=str, default=None, help="Path to custom folder to import")
    parser.add_argument("--info", action="store_true", help="Display dataset class statistics")

    args = parser.parse_args()

    if args.info:
        print_dataset_info(args.dest)
    elif args.import_zip:
        import_custom_zip(args.import_zip, dest_dir=args.dest)
    elif args.import_folder:
        import_custom_folder(args.import_folder, dest_dir=args.dest)
    elif args.dataset == "synthetic":
        generate_synthetic_dataset(dest_dir=args.dest)
    else:
        download_trashnet(dest_dir=args.dest)


if __name__ == "__main__":
    main()
