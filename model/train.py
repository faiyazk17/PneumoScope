import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# Fixed relative import for script usage
from model.dataset import PneumoDataset
from model.unet_model import UNet
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Config ---
EPOCHS = 10
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print("✅ Using device:", DEVICE)

# --- Training Function ---


def train():
    # 1. Load Dataset
    train_dataset = PneumoDataset(
        image_dir='data/images',
        mask_dir='data/masks',
        size=(256, 256)
    )
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. Model Setup
    model = UNet(n_channels=3, n_classes=1).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()  # Use sigmoid output
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses = []

    # 3. Training Loop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")

        for images, masks in loop:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)

        # Save model
        checkpoint_path = os.path.join(
            CHECKPOINT_DIR, f"unet_epoch_{epoch}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"💾 Saved checkpoint: {checkpoint_path}")
        print(f"✅ Epoch {epoch} finished | Avg Loss: {avg_loss:.4f}")

    # 4. Plot Loss Curve
    plt.figure()
    plt.plot(range(1, EPOCHS + 1), train_losses,
             marker='o', label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.savefig("training_loss_curve.png")
    plt.show()


if __name__ == "__main__":
    train()
