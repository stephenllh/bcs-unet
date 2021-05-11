from torch import nn


def init_weights(net, init_type, init_gain=0.02):
    """
    Initialize network weights.
    Parameters:
        net (network)     -- network to be initialized
        init_type (str)   -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float) -- scaling factor for normal, xavier and orthogonal.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):

            if init_type == "normal":
                nn.init.normal_(m.weight.data, 0.0, init_gain)

            elif init_type == "xavier":
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)

            elif init_type == "kaiming":
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")

            elif init_type == "orthogonal":
                nn.init.orthogonal_(m.weight.data, gain=init_gain)

            else:
                raise NotImplementedError(
                    "Initialization method [%s] is not implemented" % init_type
                )

            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

        elif (
            classname.find("BatchNorm2d") != -1
        ):  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    # print("Initialize network with %s" % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>
    return net
