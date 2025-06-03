import os
import torch
import matplotlib.pyplot as plt

from .dataset import PneumoDataset
from .unet_model import UNet

# ==== Setup ====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "checkpoints/unet_epoch_19.pth"
PROCESSED_DIR = "data/processed"
NUM_SAMPLES = 3


# ==== Load Model ====
model = UNet(n_channels=3, n_classes=1).to(DEVICE)
# Required for checkpoint loading
optimizer = torch.optim.Adam(model.parameters())


def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"✅ Loaded checkpoint from epoch {checkpoint['epoch']}")


# ==== Visualize Function ====
def visualize_predictions(model, dataset, num_samples=3):
    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

    for i in range(num_samples):
        image, true_mask = dataset[i]
        image = image.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred = model(image)
            pred_sigmoid = torch.sigmoid(pred)
            pred_thresh = (pred_sigmoid > 0.5).float()

            print(
                f"[Sample {i}] Pred min/max (raw): {pred.min().item():.4f}/{pred.max().item():.4f}")
            print(
                f"[Sample {i}] Pred min/max (sigmoid): {pred_sigmoid.min().item():.4f}/{pred_sigmoid.max().item():.4f}")
            print(
                f"[Sample {i}] % of mask predicted as 1: {pred_thresh.mean().item() * 100:.2f}%")

    # Prepare for plotting
    img_np = image.squeeze().cpu().permute(1, 2, 0).numpy()
    true_mask_np = true_mask.squeeze().cpu().numpy()
    pred_np = pred.squeeze().cpu().numpy()

    axes[i, 0].imshow(img_np)
    axes[i, 0].set_title("Image")
    axes[i, 1].imshow(true_mask_np, cmap='gray')
    axes[i, 1].set_title("True Mask")
    axes[i, 2].imshow(pred_np, cmap='gray')
    axes[i, 2].set_title("Predicted Mask")

    for ax in axes[i]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


# ==== Run ====
if __name__ == "__main__":
    dataset = PneumoDataset(processed_dir=PROCESSED_DIR)
    load_checkpoint(model, optimizer, CHECKPOINT_PATH)
    visualize_predictions(model, dataset, num_samples=NUM_SAMPLES)
