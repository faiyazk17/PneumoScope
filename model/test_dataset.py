from model.dataset import PneumoDataset
import matplotlib.pyplot as plt
import torch

# Update paths to match your folder layout
image_dir = "data/images"
mask_dir = "data/masks"

# Create dataset
dataset = PneumoDataset(image_dir=image_dir, mask_dir=mask_dir)

# Check dataset length
print(f"Total samples: {len(dataset)}")

# Load a sample
image, mask = dataset[0]

print(f"Image shape: {image.shape}")  # Should be (3, 256, 256)
print(f"Mask shape: {mask.shape}")    # Should be (1, 256, 256)

# Plot for sanity check


def show_tensor_image_mask(image_tensor, mask_tensor):
    image_np = image_tensor.permute(1, 2, 0).numpy()  # CHW -> HWC
    mask_np = mask_tensor.squeeze(0).numpy()          # Remove channel dim

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image_np)
    axes[0].set_title("Image")
    axes[1].imshow(mask_np, cmap='gray')
    axes[1].set_title("Mask")
    plt.show()


# Visual check
show_tensor_image_mask(image, mask)
