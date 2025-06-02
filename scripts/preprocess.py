# Import necessary libraries
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define paths for images and masks
image_dir = '../data/images/'  # Path to your images
mask_dir = '../data/masks/'    # Path to your segmentation masks

# List files to verify
image_files = os.listdir(image_dir)
mask_files = os.listdir(mask_dir)

# Check how many files are in each directory
# print(f'Found {len(image_files)} images and {len(mask_files)} masks.\n')


def load_single_sample(image_name, mask_name):
    """Loads one image and its corresponding mask from the dataset."""
    image_path = os.path.join(image_dir, image_name)
    mask_path = os.path.join(mask_dir, mask_name)

    # This reads the image in BGR format (default for OpenCV)
    image = cv2.imread(image_path)
    # Read mask in grayscale
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    print(f"Image shape: {image.shape}, Mask shape: {mask.shape}\n")
    return image, mask


sample_image, sample_mask = load_single_sample(image_files[0], mask_files[0])


def show_image_and_mask(image, mask):
    """Displays an image and its corresponding mask side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Show original image
    # Convert from BGR to RGB
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Chest X-ray")
    axes[0].axis('off')

    # Show segmentation mask
    axes[1].imshow(mask, cmap='gray')  # Just grayscale mask
    axes[1].set_title("Pneumothorax Mask")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


# Display the first image and its mask
# show_image_and_mask(sample_image, sample_mask)


def preprocess_image_and_mask(image, mask, size=(256, 256)):
    """
    Resizes and normalized the image and mask.
    - image: original chest x-ray (BGR)
    - mask: original segmentation mask (grayscale)
    - size: desired (height, width)
    Returns: (preprocessed image, preprocessed mask)
    """
    # Resize image and mask
    image = cv2.resize(image, size)
    mask = cv2.resize(mask, size)

    # Normalize image to [0, 1]
    image = image / 255.0

    # Binarize mask: convert to 0 and 1
    mask = (mask > 127).astype(np.float32)

    return image, mask


processed_image, processed_mask = preprocess_image_and_mask(
    sample_image, sample_mask)

print(
    f"Processed image shape: {processed_image.shape}, Processed mask shape: {processed_mask.shape}\n")
# Should be 0 and 1
print(f"Unique mask values: {np.unique(processed_mask)}\n")
