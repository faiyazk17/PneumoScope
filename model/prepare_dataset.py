import os
import torch
import cv2
from scripts.preprocess import preprocess_image_and_mask

image_dir = "data/images"
mask_dir = "data/masks"
output_dir = "data/processed"
os.makedirs(output_dir, exist_ok=True)

size = (256, 256)
image_files = sorted(os.listdir(image_dir))

for fname in image_files:
    print(f"Processing {fname}...")
    image_path = os.path.join(image_dir, fname)
    mask_path = os.path.join(mask_dir, fname)

    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    image, mask = preprocess_image_and_mask(image, mask, size)

    image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)
    mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

    torch.save((image, mask), os.path.join(
        output_dir, fname.replace("png", "pt")))

    print(f"Saved processed data for {fname} to {output_dir}")
    print(f"Processed {fname} successfully.")

print("Dataset preprocessing complete.")
