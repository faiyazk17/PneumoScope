import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image

# Loads images and masks from the specified directories and resizes them to the specified size.


def load_data(image_dir, mask_dir, img_size=(256, 256)):
    images = []
    masks = []

    image_files = os.listdir(image_dir)

    for filenames in image_files:
        # Obtaining entire path of the image and mask
        img_path = os.path.join(image_dir, filenames)
        mask_path = os.path.join(mask_dir, filenames)

        # Converting the image and mask to grayscale and resizing them
        img = Image.open(img_path).convert('L').resize(img_size)
        mask = Image.open(mask_path).convert('L').resize(img_size)

        # Converting the images and masks to numpy arrays and normalizing them
        img_array = np.array(img) / 255.0
        mask_array = np.array(mask) / 255.0

        images.append(img_array)
        masks.append(mask_array)

        # Converting the lists to numpy arrays [ML models use 3D NumPy arrays of shape [num_samples, height, width]]
        return np.array(images), np.array(masks)
