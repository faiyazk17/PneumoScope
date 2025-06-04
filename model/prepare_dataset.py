import os
import torch
import cv2
from scripts.preprocess import preprocess_image_and_mask

image_dir = "data/images"
mask_dir = "data/masks"
output_dir = "data/processed"

# Create separate folders for processed images and masks
processed_img_dir = os.path.join(output_dir, "images")
processed_mask_dir = os.path.join(output_dir, "masks")
os.makedirs(processed_img_dir, exist_ok=True)
os.makedirs(processed_mask_dir, exist_ok=True)

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

    # Save processed images and masks separately but with same filename (just .pt)
    torch.save(image, os.path.join(
        processed_img_dir, fname.replace("png", "pt")))
    torch.save(mask, os.path.join(
        processed_mask_dir, fname.replace("png", "pt")))

    print(f"Saved processed image and mask for {fname}")

print("Dataset preprocessing complete.")
