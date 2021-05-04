from torch import nn
from .utils import init_weights


class BCSUNet(nn.Module):
    """Combines simple residual upsample model and U-Net model"""

    def __init__(self, cs_model, up_model, unet_model, loss_type, init_type, **kwargs):
        super().__init__(**kwargs)
        self.cs_model = cs_model
        self.up_model = init_weights(up_model, init_type=init_type)
        self.unet_model = init_weights(unet_model, init_type=init_type)
        self.loss_type = loss_type

    def forward(self, x):
        x = self.cs_model(x)
        x = self.up_model(x)
        out = self.unet_model(x)
        return out

    def generate_images(self, real_batch, device=None):
        if device is None:
            device = self.device
        real_batch = real_batch.to("cuda")
        fake_images = self.forward(real_batch)
        return fake_images
