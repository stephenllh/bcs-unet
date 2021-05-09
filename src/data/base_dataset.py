import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class BaseDataset:
    def __init__(self, sampling_ratio: float, bcs: bool):
        if bcs:
            phi = np.load("../input/phi_block_binary.npy")
            phi = phi[: int(sampling_ratio * 16)]
            phi = torch.FloatTensor(phi)
            # print("phi_shape", phi.shape)
            self.cs_operator = BCSOperator(phi)
        else:  # TODO: prepare for CCS
            pass
            # phi = np.load("../input/________.npy")
            # self.cs_operator = CCSOperator(phi)


class BCSOperator(nn.Module):
    """
    This CNN is not trainable. It serves as a block compressive sensing operator
    by utilizing its optimized convolution operations.
    """

    def __init__(self, phi):
        super().__init__()
        self.register_buffer("phi", phi)

    def forward(self, x):
        out = F.conv2d(x, self.phi, stride=4)
        return out


class CCSOperator(nn.Module):
    """
    This CNN is not trainable. It serves as a full-image conventional compressive sensing operator
    by utilizing its optimized convolution operations.
    """

    def __init__(self, phi):
        super().__init__()
        self.register_buffer("phi", phi)

    def forward(self, x):
        assert x.shape[-1] == self.phi.shape[-1]
        out = F.conv2d(x, self.phi, stride=1)
        return out
