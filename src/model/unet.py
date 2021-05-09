from functools import partial
import torch
from torch import nn


class UNet(nn.Module):
    def __init__(
        self,
        config,
        input_nc=8,
        output_nc=1,
        num_downs=5,
    ):
        """
        Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            channels (int)  -- the number of filters in the last conv layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super().__init__()
        channels = config["channels"]
        unet_block = UnetSkipConnectionBlock(
            channels * 8,
            channels * 8,
            input_nc=None,
            submodule=None,
            norm_layer=nn.BatchNorm2d,
            innermost=True,
        )  # add the innermost layer first

        # Add intermediate layers with ngf * 8 filters
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(
                channels * 8,
                channels * 8,
                input_nc=None,
                submodule=unet_block,
                norm_layer=nn.BatchNorm2d,
                use_dropout=config["use_dropout"],
            )

        # Gradually reduce the number of filters from `channels*8` to `channels`
        unet_block = UnetSkipConnectionBlock(
            channels * 4,
            channels * 8,
            input_nc=None,
            submodule=unet_block,
            norm_layer=nn.BatchNorm2d,
        )
        unet_block = UnetSkipConnectionBlock(
            channels * 2,
            channels * 4,
            input_nc=None,
            submodule=unet_block,
            norm_layer=nn.BatchNorm2d,
        )
        unet_block = UnetSkipConnectionBlock(
            channels,
            channels * 2,
            input_nc=None,
            submodule=unet_block,
            norm_layer=nn.BatchNorm2d,
        )

        self.model = UnetSkipConnectionBlock(
            output_nc,
            channels,
            input_nc=input_nc,
            submodule=unet_block,
            outermost=True,
            norm_layer=nn.BatchNorm2d,
        )  # add the outermost layer

    def forward(self, input):
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """
    Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|

    """

    def __init__(
        self,
        outer_nc,
        inner_nc,
        input_nc=None,
        submodule=None,
        outermost=False,
        innermost=False,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
    ):
        """
        Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """

        # TODO: add spectral norm

        super().__init__()
        self.outermost = outermost
        use_bias = (
            (norm_layer.func == nn.InstanceNorm2d)
            if type(norm_layer) == partial
            else (norm_layer == nn.InstanceNorm2d)
        )  # use_bias is False if batch norm is used
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(
            input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias
        )
        downrelu = nn.LeakyReLU(0.2, inplace=True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(inplace=True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(
                inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=True
            )
            down = [downconv]
            up = [uprelu, upconv, nn.Sigmoid()]
            model = down + [submodule] + up

        elif innermost:
            upconv = nn.ConvTranspose2d(
                inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias
            )
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up

        else:  # middle layers
            upconv = nn.ConvTranspose2d(
                inner_nc * 2,
                outer_nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=use_bias,
            )
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            model = down + [submodule] + up
            if use_dropout:
                model = model + [nn.Dropout(0.5)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:  # add skip connections
            return torch.cat([x, self.model(x)], dim=1)
