import os
import yaml
import torch
import math


def voltage2pixel(y, phi, low, high):
    """
    Converts the measurement tensor / vector from voltage scale to pixel scale,
    so that the neural network can understand.
    y - measurement
    phi - block measurement matrix
    low - lowest voltage pixel
    high - highest voltage pixel
    """

    if type(y) == "numpy.ndarray":
        y = torch.from_numpy(y).float()

    phi = torch.from_numpy(phi).float()

    if y.dim() == 4:
        if y.shape[0] == 1:
            y = y.squeeze(dim=0)
        else:
            raise NotImplementedError

    if phi.dim() == 4:
        if phi.shape[1] == 1:
            phi = phi.squeeze(dim=1)
        else:
            raise NotImplementedError

    term1 = y / (high - low)
    term2 = (phi.sum(dim=(1, 2)) * low / (high - low)).unsqueeze(-1).unsqueeze(-1)
    y_pixel_scale = term1 - term2
    return y_pixel_scale


def load_config(config_path):
    with open(os.path.join(config_path)) as file:
        config = yaml.safe_load(file)
    return config


def reshape_into_block(y, sr: float, block_size=4):
    c = int(sr * block_size ** 2)
    h = w = int(math.sqrt(y.shape[0] // c))
    y = y.reshape(h, w, c)
    y = y.transpose((2, 0, 1))
    return y
