import os
import shutil
import kagglehub

# Set Kaggle credentials from user input
os.environ['KAGGLE_USERNAME'] = 'sreejithpa21112006'
os.environ['KAGGLE_KEY'] = '7f7329613e36548673653317d3fcd57e'

DEST_DIR = "dataset"
image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

KAGGLE_SOURCES = {
    "e_waste": [
        ("akshat103/e-waste-image-dataset", None),
    ],
    "organic": [
        ("techsash/waste-classification-data", "DATASET/TRAIN/O"),
        ("techsash/waste-classification-data", "DATASET/TEST/O"),
    ],
    "trash": [
        ("techsash/waste-classification-data", "DATASET/TRAIN/R"),
        ("techsash/waste-classification-data", "DATASET/TEST/R"),
    ],
}

def copy_images(src_dir, dest_cls_dir):
    copied = 0
    if not os.path.exists(src_dir):
        return 0
    for dirpath, _, filenames in os.walk(src_dir):
        for fname in filenames:
            if os.path.splitext(fname)[1].lower() in image_extensions:
                src = os.path.join(dirpath, fname)
                base, ext = os.path.splitext(fname)
                dst_name = fname
                counter = 0
                while os.path.exists(os.path.join(dest_cls_dir, dst_name)):
                    counter += 1
                    dst_name = f"{base}_{counter}{ext}"
                shutil.copy2(src, os.path.join(dest_cls_dir, dst_name))
                copied += 1
    return copied

for cls_name, sources in KAGGLE_SOURCES.items():
    cls_dir = os.path.join(DEST_DIR, cls_name)
    os.makedirs(cls_dir, exist_ok=True)
    total_copied = 0
    
    for handle, subfolder in sources:
        print(f"Downloading {handle} for {cls_name}...")
        dataset_path = kagglehub.dataset_download(handle)
        
        search_root = os.path.join(dataset_path, subfolder) if subfolder else dataset_path
        
        # If specific case subfolder doesn't match, do case insensitive search
        if subfolder and not os.path.exists(search_root):
             for root_item in os.listdir(dataset_path):
                 if root_item.lower() == subfolder.split('/')[0].lower():
                     search_root = os.path.join(dataset_path, root_item)
                     for remaining_part in subfolder.split('/')[1:]:
                         found = False
                         for sub_item in os.listdir(search_root):
                             if sub_item.lower() == remaining_part.lower():
                                 search_root = os.path.join(search_root, sub_item)
                                 found = True
                                 break
                         if not found:
                             break
        
        copied = copy_images(search_root, cls_dir)
        total_copied += copied
        
    print(f"-> {cls_name}: {total_copied} images copied.")

print("Done downloading new classes!")
