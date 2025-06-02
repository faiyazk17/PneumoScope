import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # <-- import TensorBoard writer
from .dataset import PneumoDataset
from .unet_model import UNet
import os
from tqdm import tqdm

# Hyperparameters and config
EPOCHS = 20
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def save_checkpoint(model, optimizer, epoch):
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"unet_epoch_{epoch}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch}")


def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Checkpoint loaded: epoch {checkpoint['epoch']}")
    return checkpoint['epoch']


def train(resume_checkpoint=None):
    train_dataset = PneumoDataset(
        image_dir='data/images',
        mask_dir='data/masks',
        size=(256, 256))
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = UNet(n_channels=3, n_classes=1).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    start_epoch = 1
    if resume_checkpoint:
        start_epoch = load_checkpoint(model, optimizer, resume_checkpoint) + 1

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir="runs/pneumoscope_experiment")

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")

        for batch_idx, (images, masks) in enumerate(loop):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

            # Log loss per batch (optional)
            global_step = (epoch - 1) * len(train_loader) + batch_idx
            writer.add_scalar("Loss/train_batch", loss.item(), global_step)

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch} finished with avg loss: {avg_loss:.4f}")

        # Log avg epoch loss
        writer.add_scalar("Loss/train_epoch", avg_loss, epoch)

        save_checkpoint(model, optimizer, epoch)

    writer.close()


if __name__ == "__main__":
    train(resume_checkpoint=None)
