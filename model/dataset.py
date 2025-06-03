import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from scripts.preprocess import preprocess_image_and_mask


class PneumoDataset(Dataset):
    # def __init__(self, image_dir, mask_dir, size=(256, 256)):
    #     self.image_dir = image_dir
    #     self.mask_dir = mask_dir
    #     self.size = size

    #     self.image_files = sorted(os.listdir(image_dir))
    #     self.mask_files = sorted(os.listdir(mask_dir))

    # def __len__(self):
    #     return len(self.image_files)

    # def __getitem__(self, idx):
    #     image_path = os.path.join(self.image_dir, self.image_files[idx])
    #     mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

    #     image = cv2.imread(image_path)
    #     mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    #     image, mask = preprocess_image_and_mask(image, mask, self.size)

    #     # Convert to PyTorch tensors and rearrange dimentions to CxHxW format (PyTorch format)
    #     image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)
    #     mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

    #     return image, mask

    def __init__(self, processed_dir):
        self.files = sorted(os.listdir(processed_dir))
        self.processed_dir = processed_dir

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return torch.load(os.path.join(self.processed_dir, self.files[idx]))
