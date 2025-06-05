import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from model.balanced_sampler import BalancedSampler
from model.positive_sampler import PositiveOversampleSampler

from .dataset import PneumoDataset
from .unet_model import UNet

from torch.utils.data import random_split

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
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(DiceBCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, inputs, targets):
        smooth = 1.0
        inputs_sig = torch.sigmoid(inputs)
        inputs_flat = inputs_sig.view(-1)
        targets_flat = targets.view(-1)

        intersection = (inputs_flat * targets_flat).sum()
        dice = (2. * intersection + smooth) / \
            (inputs_flat.sum() + targets_flat.sum() + smooth)

        bce_loss = self.bce(inputs, targets)
        dice_loss = 1 - dice
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


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


def dice_coeff(inputs, targets, smooth=1.0):
    inputs = torch.sigmoid(inputs)
    inputs = (inputs > 0.5).float()
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    intersection = (inputs * targets).sum()
    return (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)


def load_best_model(model):
    best_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
        print(f"Best model loaded from {best_model_path}")
    else:
        print("Best model not found. Ensure training with early stopping was completed.")


# ==== Training Function ====
def train(resume_checkpoint=None):
    # Dataset & DataLoader
    train_dataset = PneumoDataset(
        processed_dir_img='data/processed/images',
        processed_dir_mask='data/processed/masks'
    )

    # Split: 90% train, 10% val
    val_size = int(0.1 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(
        train_dataset, [train_size, val_size])

    # Calculate weights for balanced sampling
    # labels = []
    # for _, mask in train_dataset:
    #     label = 1 if mask.sum() > 0 else 0  # 1 = positive case, 0 = empty mask
    #     labels.append(label)

    # class_counts = [labels.count(0), labels.count(1)]
    # class_weights = [1.0 / c for c in class_counts]
    # sample_weights = [class_weights[label] for label in labels]
    # sampler = WeightedRandomSampler(
    #     sample_weights, num_samples=len(sample_weights), replacement=True)

    labels = [train_dataset[i][1].sum().item(
    ) > 0 for i in train_dataset.indices]
    sampler = PositiveOversampleSampler(labels, num_samples=len(train_dataset))

    print(f"Found {len(train_dataset)} samples")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=12,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,   # Fewer workers are fine here
        pin_memory=True
    )

    # Quick check: what % of masks in training set are completely empty?
    empty_masks = 0
    for _, mask in train_dataset:
        if mask.sum() == 0:
            empty_masks += 1
    print(f"{empty_masks}/{len(train_dataset)} masks are completely empty ({(empty_masks / len(train_dataset)) * 100:.2f}%)")

    # Model, loss, optimizer, scaler
    model = UNet(n_channels=3, n_classes=1).to(DEVICE)
    criterion = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    # Resume if needed
    start_epoch = 1
    best_val_dice = 0.0
    patience = 5
    epochs_without_improvement = 0

    if resume_checkpoint == "best":
        load_best_model(model)
    elif resume_checkpoint:
        start_epoch = load_checkpoint(model, optimizer, resume_checkpoint) + 1

    # TensorBoard
    writer = SummaryWriter(log_dir="runs/pneumoscope_experiment")

    # Training Loop
    for epoch in range(start_epoch, EPOCHS + 1):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        running_dice = 0.0
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
            batch_dice = dice_coeff(outputs, masks).item()
            running_dice += batch_dice

            loop.set_postfix(loss=float(loss), dice=batch_dice)

            if batch_idx % 10 == 0:
                global_step = (epoch - 1) * len(train_loader) + batch_idx
                writer.add_scalar("Loss/train_batch", float(loss), global_step)
                writer.add_scalar("Dice/train_batch", batch_dice, global_step)

        avg_loss = running_loss / len(train_loader)
        avg_dice = running_dice / len(train_loader)
        print(
            f"Epoch {epoch} finished with avg loss: {avg_loss:.4f}, avg dice: {avg_dice:.4f}")
        writer.add_scalar("Loss/train_epoch", avg_loss, epoch)
        writer.add_scalar("Dice/train_epoch", avg_dice, epoch)

        save_checkpoint(model, optimizer, epoch)

        epoch_duration = (time.time() - start_time) / 60
        print(f"Epoch {epoch} duration: {epoch_duration:.2f} minutes")

        # === Validation ===
        model.eval()
        val_loss = 0.0
        val_dice = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(DEVICE, non_blocking=True)
                masks = masks.to(DEVICE, non_blocking=True)

                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += float(loss)
                val_dice += dice_coeff(outputs, masks).item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)

        print(
            f"[Validation] Epoch {epoch} - Loss: {avg_val_loss:.4f}, Dice: {avg_val_dice:.4f}")
        writer.add_scalar("Loss/val_epoch", avg_val_loss, epoch)
        writer.add_scalar("Dice/val_epoch", avg_val_dice, epoch)

        # Early stopping logic
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            epochs_without_improvement = 0
            torch.save(model.state_dict(), os.path.join(
                CHECKPOINT_DIR, "best_model.pth"))
            print(f"New best model saved with Dice: {best_val_dice:.4f}")
        else:
            epochs_without_improvement += 1
            print(
                f"No improvement in validation Dice for {epochs_without_improvement} epoch(s)")

        if epochs_without_improvement >= patience:
            print(
                f"Early stopping at epoch {epoch} due to no improvement in Dice.")
            break

    writer.close()


# ==== Run Entry Point ====
if __name__ == "__main__":
    train(resume_checkpoint=None)
