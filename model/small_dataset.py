import os
import shutil

# Source folder with all processed .pt files
src_dir = 'data/processed'
# Target folder with fewer samples
dst_dir = 'data/processed_small'
os.makedirs(dst_dir, exist_ok=True)

# Number of samples to copy
N = 100

# Copy first N files
for fname in sorted(os.listdir(src_dir))[:N]:
    shutil.copy(os.path.join(src_dir, fname), os.path.join(dst_dir, fname))

print(f"Copied {N} samples to {dst_dir}")
