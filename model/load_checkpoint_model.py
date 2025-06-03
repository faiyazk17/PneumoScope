import torch
from .unet_model import UNet  # adjust the path if needed


def load_model(checkpoint_path, device='cuda'):
    model = UNet(n_channels=3, n_classes=1).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded from {checkpoint_path}")
    return model


model = load_model("checkpoints/unet_epoch_19.pth")
