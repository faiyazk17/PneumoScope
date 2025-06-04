import os
import shutil

src_img_dir = 'data/processed/images'
src_mask_dir = 'data/processed/masks'

dst_img_dir = 'data/processed_small/images'
dst_mask_dir = 'data/processed_small/masks'

os.makedirs(dst_img_dir, exist_ok=True)
os.makedirs(dst_mask_dir, exist_ok=True)

# Only get training files
all_img_files = sorted([f for f in os.listdir(src_img_dir) if 'train' in f])
copied = 0

for fname in all_img_files:
    if os.path.exists(os.path.join(src_mask_dir, fname)):
        shutil.copy(os.path.join(src_img_dir, fname),
                    os.path.join(dst_img_dir, fname))
        shutil.copy(os.path.join(src_mask_dir, fname),
                    os.path.join(dst_mask_dir, fname))
        copied += 1
    if copied >= 100:
        break

print(f"Copied {copied} training samples to smaller processed dataset.")
