import matplotlib.pyplot as plt
import numpy as np
import torch


def visualize_predictions(model, dataloader, device, threshold=0.3, num_samples=5):
    model.eval()
    count = 0

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            outputs = torch.sigmoid(model(images)).cpu().numpy()
            images = images.cpu().numpy()
            masks = masks.cpu().numpy()

            for i in range(images.shape[0]):
                pred_mask = outputs[i, 0] > threshold
                true_mask = masks[i, 0]
                image = images[i].transpose(1, 2, 0)  # CxHxW to HxWxC

                plt.figure(figsize=(12, 4))
                plt.subplot(1, 3, 1)
                plt.imshow(image, cmap='gray')
                plt.title('Image')

                plt.subplot(1, 3, 2)
                plt.imshow(true_mask, cmap='gray')
                plt.title('Ground Truth Mask')

                plt.subplot(1, 3, 3)
                plt.imshow(pred_mask, cmap='gray')
                plt.title(f'Predicted Mask (th={threshold})')

                plt.show()

                count += 1
                if count >= num_samples:
                    return
