from torch import nn
from .upsamplenet import UpsampleNet
from .unet import UNet
from .utils import init_weights


class BCSUNet(nn.Module):
    """Combines simple residual upsample model and U-Net model"""

    def __init__(self, net_config):
        super().__init__()
        config = net_config
        upsamplenet = UpsampleNet(
            sampling_ratio=config["sampling_ratio"], config=config["upsamplenet"]
        )
        self.upsamplenet = init_weights(
            upsamplenet, init_type=config["upsamplenet"]["init_type"]
        )
        unet = UNet(config["unet"])
        self.unet = init_weights(unet, init_type=config["unet"]["init_type"])

    def forward(self, x):
        out1 = self.upsamplenet(x)
        out2 = self.unet(out1)
        return out1, out2
