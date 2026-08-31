import os
import shutil
import random
import kagglehub

random.seed(42)

# 1. Clear out the wrong "trash" (Recyclable) folder
print("Clearing wrong trash folder...")
trash_dir = os.path.join("dataset", "trash")
if os.path.exists(trash_dir):
    shutil.rmtree(trash_dir)
os.makedirs(trash_dir, exist_ok=True)

# 2. Download proper "trash" from mostafaabla/garbage-classification
print("Downloading proper trash images...")
dataset_path = kagglehub.dataset_download("mostafaabla/garbage-classification")

search_root = None
for root, dirs, files in os.walk(dataset_path):
    if "trash" in [d.lower() for d in dirs]:
        for d in dirs:
            if d.lower() == "trash":
                search_root = os.path.join(root, d)
                break
        break

if search_root:
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    for fname in os.listdir(search_root):
        if os.path.splitext(fname)[1].lower() in image_extensions:
            shutil.copy2(os.path.join(search_root, fname), os.path.join(trash_dir, fname))
    print(f"Copied {len(os.listdir(trash_dir))} proper trash images.")

# 3. Subsample folders down to 1000 images to balance the dataset
def subsample_folder(folder_path, max_images=1000):
    if not os.path.exists(folder_path):
        return
    
    # We want to keep synthetic images (they start with syn_) and randomly sample real ones
    files = os.listdir(folder_path)
    syn_files = [f for f in files if f.startswith("syn_")]
    real_files = [f for f in files if not f.startswith("syn_")]
    
    if len(files) > max_images:
        print(f"Subsampling {folder_path} from {len(files)} to {max_images}...")
        num_real_to_keep = max_images - len(syn_files)
        
        real_files_to_delete = random.sample(real_files, len(real_files) - num_real_to_keep)
        for fname in real_files_to_delete:
            os.remove(os.path.join(folder_path, fname))
        print(f"-> {folder_path} now has {len(os.listdir(folder_path))} images.")

subsample_folder(os.path.join("dataset", "e_waste"), max_images=1000)
subsample_folder(os.path.join("dataset", "organic"), max_images=1000)
subsample_folder(os.path.join("dataset", "trash"), max_images=1000)

print("Dataset fixed and balanced!")
