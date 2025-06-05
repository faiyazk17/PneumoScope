# evaluate_thresholds.py

import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from model.unet_model import UNet
from model.dataset import PneumoDataset
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "./checkpoints/unet_epoch_20.pth"  # Update if needed
DATA_DIR_IMG = "data/processed/images"
DATA_DIR_MASK = "data/processed/masks"
BATCH_SIZE = 8


def dice_coeff(preds, targets, smooth=1.):
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    return (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)


def load_model():
    model = UNet(n_channels=3, n_classes=1).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def get_val_loader():
    dataset = PneumoDataset(DATA_DIR_IMG, DATA_DIR_MASK)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)


if __name__ == "__main__":
    model = load_model()
    val_loader = get_val_loader()

    print("Model and validation loader ready.")

    thresholds = [i / 10 for i in range(1, 10)]  # 0.1 to 0.9
    best_threshold = 0.5
    best_dice = 0.0

    with torch.no_grad():
        for thresh in thresholds:
            dices = []
            for images, masks in tqdm(val_loader, desc=f"Threshold {thresh:.1f}"):
                images = images.to(DEVICE, non_blocking=True)
                masks = masks.to(DEVICE, non_blocking=True)

                outputs = model(images)
                probs = torch.sigmoid(outputs)

                preds = (probs > thresh).float()
                batch_dice = dice_coeff(preds, masks)
                dices.append(batch_dice.item())

            avg_dice = sum(dices) / len(dices)
            print(f"Threshold {thresh:.1f}: Average Dice = {avg_dice:.4f}")

            if avg_dice > best_dice:
                best_dice = avg_dice
                best_threshold = thresh

    print(f"\nBest threshold: {best_threshold:.1f} with Dice: {best_dice:.4f}")
