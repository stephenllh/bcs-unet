from torch import nn
from .upsamplenet import UpsampleNet, ReshapeNet
from .unet import UNet
from .utils import init_weights


class BCSUNet(nn.Module):
    """Combines simple residual upsample model and U-Net model"""

    def __init__(self, config):
        super().__init__()
        upsamplenet = UpsampleNet(
            sampling_ratio=config["sampling_ratio"],
            upsamplenet_config=config["net"]["upsamplenet"],
        )
        self.upsamplenet = init_weights(
            upsamplenet, init_type=config["net"]["upsamplenet"]["init_type"]
        )
        unet = UNet(config["net"]["unet"], input_nc=8)
        self.unet = init_weights(unet, init_type=config["net"]["unet"]["init_type"])

    def forward(self, x):
        out1a, out1b = self.upsamplenet(x)
        out2 = self.unet(out1a)
        return out1b, out2
