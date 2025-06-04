import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from scripts.preprocess import preprocess_image_and_mask


import os
import torch
from torch.utils.data import Dataset


class PneumoDataset(torch.utils.data.Dataset):
    def __init__(self, processed_dir_img, processed_dir_mask):
        self.processed_dir_img = processed_dir_img
        self.processed_dir_mask = processed_dir_mask

        # Filter files to only those containing "train" in the filename
        all_files = sorted(os.listdir(self.processed_dir_img))
        self.files = [f for f in all_files if "train" in f]

        # Precompute mask labels: 1 if pneumothorax present, else 0
        self.labels = []
        for fname in self.files:
            mask = torch.load(os.path.join(self.processed_dir_mask, fname))
            self.labels.append(int(mask.sum() > 0))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = torch.load(os.path.join(
            self.processed_dir_img, self.files[idx]))
        mask = torch.load(os.path.join(
            self.processed_dir_mask, self.files[idx]))
        return image, mask

    def get_labels(self):
        return self.labels
