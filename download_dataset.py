"""
OpticBin — Dataset Downloader (TrashNet)
========================================
Downloads and extracts the 5-class waste dataset into dataset/
"""

import os
import shutil
import sys
import urllib.request
import zipfile

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

DATASET_URL = "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip"
DEST_DIR = "dataset"
ZIP_PATH = "trashnet.zip"


def download_trashnet():
    os.makedirs(DEST_DIR, exist_ok=True)

    print(f"📥 Downloading TrashNet dataset...")
    try:
        urllib.request.urlretrieve(DATASET_URL, ZIP_PATH)
        print("✓ Download complete.")

        print("📦 Extracting dataset...")
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall("dataset_temp")

        extracted_root = os.path.join("dataset_temp", "dataset-resized")
        if os.path.exists(extracted_root):
            for folder in os.listdir(extracted_root):
                src = os.path.join(extracted_root, folder)
                dst = os.path.join(DEST_DIR, folder)
                if os.path.isdir(src):
                    if os.path.exists(dst):
                        shutil.rmtree(dst)
                    shutil.move(src, dst)

            shutil.rmtree("dataset_temp")

        if os.path.exists(ZIP_PATH):
            os.remove(ZIP_PATH)

        # Remove trash folder if present (as requested)
        trash_dir = os.path.join(DEST_DIR, "trash")
        if os.path.exists(trash_dir):
            shutil.rmtree(trash_dir)

        print(f"✅ Dataset successfully prepared in '{DEST_DIR}/' (5 classes)")
    except Exception as e:
        print(f"❌ Could not auto-download: {e}")
        print(f"💡 You can manually place images into subfolders inside '{DEST_DIR}/':")
        print("   dataset/glass, dataset/paper, dataset/cardboard, dataset/plastic, dataset/metal")


if __name__ == "__main__":
    download_trashnet()
