import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    A block with two convolutional layers, each followed by ReLU activation.
    Keeps image size the samme (padding=1).
    """

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

# Testing the above code
# x = torch.randn(1, 3, 256, 256)  # Batch of 1 RGB image
# conv = DoubleConv(3, 64)
# y = conv(x)
# print(y.shape)  # Should be [1, 64, 256, 256]


class Down(nn.Module):
    """
    Downscaling with maxpool followed by DoubleConv.
    """

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)


# Testing the above code
# x = torch.randn(1, 64, 256, 256)  # Simulating a previous layer output
# down = Down(64, 128)
# y = down(x)
# print(y.shape)  # Should be [1, 128, 128, 128]

class Up(nn.Module):
    """
    Upscaling then double conv. Uses skip connections from encoder.
    """

    def __init__(self, in_channels, out_channels, skip_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels=out_channels + skip_channels,  # Concatenated channels
                               out_channels=out_channels)

    def forward(self, x1, x2):
        # x1 is the upsampled feature map, x2 is the skip connection from encoder
        x1 = self.up(x1)

        # Ensure dimentions match due to possible rounding
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)  # Concatenate along channels

        return self.conv(x)


# Testing the above code
# x1 = torch.randn(1, 256, 64, 64)   # from decoder
# x2 = torch.randn(1, 256, 128, 128)  # from encoder
# up = Up(in_channels=256, out_channels=128, skip_channels=256)
# y = up(x1, x2)
# print(y.shape)  # Should be [1, 128, 128, 128]
