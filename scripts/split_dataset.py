import os
import random
import shutil
from pathlib import Path

def split_dataset(base_dir, train_ratio=0.8):
    """
    Split the dataset into train and validation sets.
    Move files from dataset/images/train and dataset/labels/train 
    to dataset/images/val and dataset/labels/val.
    """
    images_train_dir = base_dir / "dataset/images/train"
    labels_train_dir = base_dir / "dataset/labels/train"
    images_val_dir = base_dir / "dataset/images/val"
    labels_val_dir = base_dir / "dataset/labels/val"
    
    # Create val directories if they don't exist
    os.makedirs(images_val_dir, exist_ok=True)
    os.makedirs(labels_val_dir, exist_ok=True)
    
    # Get all image files in train
    image_files = sorted([f for f in os.listdir(images_train_dir) if f.endswith(".jpg")])
    
    # Randomly shuffle
    random.seed(42) # For reproducibility
    random.shuffle(image_files)
    
    # Calculate split index
    split_idx = int(len(image_files) * (1 - train_ratio))
    val_files = image_files[:split_idx]
    
    print(f"Splitting {len(image_files)} images into {len(image_files) - split_idx} train and {split_idx} val...")
    
    moved_count = 0
    for img_file in val_files:
        label_file = img_file.replace(".jpg", ".txt")
        
        src_img = images_train_dir / img_file
        dst_img = images_val_dir / img_file
        src_lbl = labels_train_dir / label_file
        dst_lbl = labels_val_dir / label_file
        
        # Move image
        if os.path.exists(src_img):
            shutil.move(src_img, dst_img)
            
        # Move label (if exists)
        if os.path.exists(src_lbl):
            shutil.move(src_lbl, dst_lbl)
            moved_count += 1
            
    print(f"✅ Moved {len(val_files)} images and {moved_count} labels to validation set.")

if __name__ == "__main__":
    base_dir = Path("c:/Users/ADMIN/Documents/Fall_Detection_2")
    split_dataset(base_dir)
