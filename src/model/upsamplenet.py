from torch import nn
from torch.nn import functional as F
from model.layers import SNConv2d


class ReshapeNet(nn.Module):
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


class UpsampleNet(nn.Module):
    def __init__(self, sampling_ratio, upsamplenet_config):
        super().__init__()
        kernel_size = 4
        first_out_channels = int(sampling_ratio * kernel_size ** 2)
        config = upsamplenet_config

        self.up1 = UpResBlock(
            in_channels=first_out_channels,
            out_channels=config["out_channels_1"],
            middle_channels=None,
            upsample=True,
            use_transpose_conv=config["use_transpose_conv"],
            spectral_norm=config["spectral_norm"],
        )

        self.up2 = UpResBlock(
            in_channels=config["out_channels_1"],
            out_channels=config["out_channels_2"],
            middle_channels=None,
            upsample=True,
            use_transpose_conv=config["use_transpose_conv"],
            spectral_norm=config["spectral_norm"],
        )

    def forward(self, x):
        x = self.up1(x)
        out = self.up2(x)  # passed to UNet
        return out


class UpResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        middle_channels=None,
        upsample=True,
        use_transpose_conv=False,
        norm_type="instance",
        spectral_norm=True,
        init_type="xavier",
    ):

        super().__init__()
        self.upsample = upsample
        self.use_transpose_conv = use_transpose_conv

        if middle_channels is None:
            middle_channels = out_channels

        if use_transpose_conv:
            assert upsample is True
            self.conv1 = nn.ConvTranspose2d(
                in_channels,
                middle_channels,
                kernel_size=2,
                stride=2,
                padding=1,
                bias=False,
            )
            self.conv2 = nn.ConvTranspose2d(
                middle_channels,
                out_channels,
                kernel_size=2,
                stride=2,
                padding=1,
                bias=False,
            )

        else:  # if transpose conv is not used.
            # The `_residual_block` method will decide whether or not it upsamples depending on `upsample == True/False`
            conv = SNConv2d if spectral_norm else nn.Conv2d
            self.conv1 = conv(
                in_channels,
                middle_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            self.conv2 = conv(
                middle_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )

        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def _upsample(self, x, conv_layer):
        if self.use_transpose_conv:
            return conv_layer(x)
        else:
            return conv_layer(
                F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            )

    def _residual_block(self, x):
        x = self._upsample(x, self.conv1) if self.upsample else self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        out = self.relu(x)
        return out

    def _shortcut(self, x):
        return self._upsample(x, self.conv1) if self.upsample else self.conv1(x)

    def forward(self, x):
        return self._residual_block(x) + self._shortcut(x)
