import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model.unet_model import UNet
from model.dataset import PneumoDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dice_coeff(pred, target, smooth=1.0):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def visualize_predictions(model_path, dataset_path, num_samples=5):
    # Load model
    model = UNet(n_channels=3, n_classes=1).to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    dataset = PneumoDataset(processed_dir_img=f"{dataset_path}/images",
                            processed_dir_mask=f"{dataset_path}/masks")
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    with torch.no_grad():
        for i, (image, mask) in enumerate(loader):
            if i >= num_samples:
                break

            image = image.to(DEVICE)
            output = model(image)
            pred = torch.sigmoid(output).cpu().squeeze().numpy()
            pred_mask = (pred > 0.5).astype(float)

            img_np = image.cpu().squeeze().permute(1, 2, 0).numpy()
            true_mask = mask.cpu().squeeze().numpy()

            # Dice score for this sample
            dice_score = dice_coeff(torch.tensor(
                pred_mask), torch.tensor(true_mask))

            # Plot
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            axs[0].imshow(img_np, cmap='gray')
            axs[0].set_title("Image")
            axs[1].imshow(true_mask, cmap='gray')
            axs[1].set_title("Ground Truth Mask")
            axs[2].imshow(pred_mask, cmap='gray')
            axs[2].set_title(f"Predicted Mask\nDice: {dice_score:.4f}")

            for ax in axs:
                ax.axis('off')

            plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True,
                        help="Path to checkpoint .pth file")
    parser.add_argument("--dataset_path", default="data/processed_small",
                        help="Processed dataset directory")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of samples to visualize")
    args = parser.parse_args()

    visualize_predictions(args.model_path, args.dataset_path, args.num_samples)
