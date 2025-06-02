import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from scripts.preprocess import preprocess_image_and_mask


class PneumoDataset(Dataset):
    def __init__(self, image_dir, mask_dir, size=(256, 256)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.size = size

        self.image_files = sorted(os.listdir(image_dir))
        self.mask_files = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path
