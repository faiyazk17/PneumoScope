import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from .dataset import PneumoDataset
from .unet_model import UNet

torch.backends.cudnn.benchmark = True

# ==== Config ====
EPOCHS = 20
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print("Using device:", DEVICE)


# ==== Custom Dice + BCE Loss ====
class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        smooth = 1.0
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / \
            (inputs.sum() + targets.sum() + smooth)

        return 1 - dice + self.bce(inputs, targets)


# ==== Checkpoint Functions ====
def save_checkpoint(model, optimizer, epoch):
    path = os.path.join(CHECKPOINT_DIR, f"unet_epoch_{epoch}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)
    print(f"Checkpoint saved at epoch {epoch}")


def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Checkpoint loaded: epoch {checkpoint['epoch']}")
    return checkpoint['epoch']


# ==== Training Function ====
def train(resume_checkpoint=None):
    # Dataset & DataLoader
    train_dataset = PneumoDataset(processed_dir='data/processed')
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=12,
        pin_memory=True,
        persistent_workers=True
    )

    # Model, loss, optimizer, scaler
    model = UNet(n_channels=3, n_classes=1).to(DEVICE)
    criterion = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    # Resume if needed
    start_epoch = 1
    if resume_checkpoint:
        start_epoch = load_checkpoint(model, optimizer, resume_checkpoint) + 1

    # TensorBoard
    writer = SummaryWriter(log_dir="runs/pneumoscope_experiment")

    # Training Loop
    for epoch in range(start_epoch, EPOCHS + 1):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")

        for batch_idx, (images, masks) in enumerate(loop):
            images = images.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            with autocast(enabled=torch.cuda.is_available()):
                outputs = model(images)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0)  # Gradient clipping
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss)
            loop.set_postfix(loss=float(loss))

            if batch_idx % 10 == 0:
                global_step = (epoch - 1) * len(train_loader) + batch_idx
                writer.add_scalar("Loss/train_batch", float(loss), global_step)

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch} finished with avg loss: {avg_loss:.4f}")
        writer.add_scalar("Loss/train_epoch", avg_loss, epoch)

        save_checkpoint(model, optimizer, epoch)

        epoch_duration = (time.time() - start_time) / 60
        print(f"Epoch {epoch} duration: {epoch_duration:.2f} minutes")

    writer.close()


# ==== Run Entry Point ====
if __name__ == "__main__":
    train(resume_checkpoint=None)
