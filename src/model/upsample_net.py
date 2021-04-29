import torch
from torch import nn
from torch.nn import functional as F
from model.utils import get_norm_layer
from model.layers import SNConv2d


class LearnableCompressiveSensingNet(nn.Module):
    def __init__(self, sampling_ratio, kernel_size):
        super().__init__()
        first_out_channels = int(sampling_ratio * kernel_size ** 2)
        self.measurement = nn.Conv2d(
            1, first_out_channels, kernel_size, stride=kernel_size, bias=False
        )

    def forward(self, x):
        return self.measurement(x)


class UpResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        middle_channels=None,
        upsample=True,
        use_transpose_conv=False,
        learnable_sc=True,
        norm_type="instance",
        spectral_norm=True,
        init_type="xavier",
    ):

        super().__init__()
        self.upsample = upsample
        self.use_transpose_conv = use_transpose_conv
        self.learnable_sc = learnable_sc

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

        else:  # if transpose conv is not used. Whether or not it upsamples (see upsample == True/False),
            # it is done in `_residual_block` method
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

        norm = get_norm_layer(norm_type)
        self.bn1 = norm(middle_channels)
        self.bn2 = norm(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def _upsample(self, x, conv_layer):
        if self.use_transpose_conv:
            return conv_layer(x)
        else:
            return conv_layer(
                F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            )

    def _residual_block(self, x):  # bn - relu - conv - bn - relu - conv
        x = self.bn1(x)
        x = self.relu(x)
        x = self._upsample(x, self.conv1) if self.upsample else self.conv1(x)
        x = self.relu(x)
        x = self.bn2(x)
        out = self.conv2(x)
        return out

    def _shortcut(self, x):
        if self.learnable_sc:
            return self._upsample(x, self.conv1) if self.upsample else self.conv1(x)
        else:
            return x

    def forward(self, x):
        return self._residual_block(x) + self._shortcut(x)


class Decoder(nn.Module):
    """The decoder of the whole architecture. It uses UpResBlock"""

    def __init__(
        self,
        sampling_ratio,
        kernel_size=4,
        channels_list=(16, 8),
        transpose_conv=False,
        learnable_sc=True,
        spectral_norm=True,
    ):
        super().__init__()

        first_out_channels = int(sampling_ratio * kernel_size ** 2)

        self.up1 = UpResBlock(
            in_channels=first_out_channels,
            out_channels=channels_list[0],
            middle_channels=None,
            upsample=True,
            use_transpose_conv=False,
            learnable_sc=True,
            spectral_norm=True,
        )

        self.up2 = UpResBlock(
            in_channels=channels_list[0],
            out_channels=channels_list[1],
            middle_channels=None,
            upsample=True,
            use_transpose_conv=False,
            learnable_sc=True,
            spectral_norm=True,
        )

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        return torch.tanh(x)
