import torch
from torch import nn


class SCSNetInit(nn.Module):
    """The "initial reconstruction network" of SCSNet"""

    def __init__(self, in_channels, block_size=4):
        super().__init__()
        self.block_size = block_size
        self.conv = nn.Conv2d(in_channels, block_size ** 2, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        out = self._permute(x)
        return out

    def _permute(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.view(B, H, W, self.block_size, self.block_size)
        x = x.permute(0, 1, 3, 2, 4).contiguous()
        out = x.view(-1, 1, H * self.block_size, W * self.block_size)
        return out


class SCSNetDeep(nn.Module):
    """The "deep reconstruction network" of SCSNet"""

    def __init__(self):
        super().__init__()
        middle_convs = [
            ConvBlock(in_channels=32, out_channels=32, kernel_size=3) for _ in range(13)
        ]
        self.convs = nn.Sequential(
            ConvBlock(in_channels=1, out_channels=128, kernel_size=3),
            ConvBlock(in_channels=128, out_channels=32, kernel_size=3),
            *middle_convs,
            ConvBlock(in_channels=32, out_channels=128, kernel_size=3),
            nn.Conv2d(
                in_channels=128,
                out_channels=1,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
        )

    def forward(self, x):
        out = self.convs(x)
        return torch.sigmoid(x + out)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=True,
        )
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        out = self.relu(x)
        return out
