import torch
from torch import nn


class ReconNet(nn.Module):
    def __init__(self, num_measurements, img_dim):
        super().__init__()
        self.img_dim = img_dim
        self.linear = nn.Linear(num_measurements, img_dim * img_dim)
        self.bn = nn.BatchNorm1d(img_dim * img_dim)
        self.convs = nn.Sequential(
            ConvBlock(in_channels=1, out_channels=64, kernel_size=11),
            ConvBlock(in_channels=64, out_channels=32, kernel_size=1),
            ConvBlock(in_channels=32, out_channels=1, kernel_size=7),
            ConvBlock(in_channels=1, out_channels=64, kernel_size=11),
            ConvBlock(in_channels=64, out_channels=32, kernel_size=1),
        )
        self.final_conv = nn.Conv2d(32, 1, kernel_size=7, padding=7 // 2)

    def forward(self, x):
        if x.dim() == 4:  # BCS
            x = x.view(x.shape[0], -1)
            x = x.unsqueeze(dim=1)
        x = self.linear(x)
        x = x.view(-1, 1, self.img_dim, self.img_dim)
        x = self.convs(x)
        x = self.final_conv(x)
        out = torch.sigmoid(x)
        return out


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
