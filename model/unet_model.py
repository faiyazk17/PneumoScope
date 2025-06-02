import torch
import torch.nn as nn
from .unet_parts import DoubleConv, Down, Up


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()

        self.inc = DoubleConv(n_channels, 64)  # Initial convolution

        # Encoder path (Downsampling)
        self.down1 = Down(64, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # Decoder path (Upsampling)
        self.up1 = Up(1024, 512, skip_channels=512)
        self.up2 = Up(512, 256, skip_channels=256)
        self.up3 = Up(256, 128, skip_channels=128)
        self.up4 = Up(128, 64, skip_channels=64)

        # Final output layer: 1x1 convolution to get the desired number of classes
        self.out_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)  # Initial convolution, output size: [batch, 64, H, W]

        # Encoder path
        x2 = self.down1(x1)  # output size: [batch, 64, H/2, W/2]
        x3 = self.down2(x2)  # output size: [batch, 128, H/4, W/4]
        x4 = self.down3(x3)  # output size: [batch, 256, H/8, W/8]
        x5 = self.down4(x4)  # output size: [batch, 512, H/16, W/16]

        # Bottleneck
        # output size: [batch, 1024, H/16, W/16]
        bottleneck = self.bottleneck(x5)

        # Decoder path
        # combines bottleneck & x4 skip, output size: [batch, 512, H/8, W/8]
        up1 = self.up1(bottleneck, x5)
        # combines upsampled x & x3 skip, output size: [batch, 256, H/4, W/4]
        up2 = self.up2(up1, x4)
        # combines upsampled x & x2 skip, output size: [batch, 128, H/2, W/2]
        up3 = self.up3(up2, x3)
        # combines upsampled x & x1 skip, output size: [batch, 64, H, W]
        up4 = self.up4(up3, x1)

        # Final output layer (1x1 conv)
        output = self.out_conv(up4)  # output size: [batch, n_classes, H, W]

        return output


# # Create a dummy input tensor with batch=1, 3 channels (RGB), 256x256 pixels
# dummy_input = torch.randn(1, 3, 256, 256)

# # Create the model instance, assuming binary segmentation (output channels=1)
# model = UNet(n_channels=3, n_classes=1)

# # Forward pass
# output = model(dummy_input)

# # Print output shape, should be (1, 1, 256, 256)
# print("Output shape:", output.shape)

# # Simple assertion to check output shape is correct
# assert output.shape == (1, 1, 256, 256), "Output shape is incorrect"

# print("UNet forward pass test passed!")
