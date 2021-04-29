import numpy as np
import torch
import torchvision
from data.dataset import BCSOperator


class ShiftedMNISTDataset:
    def __init__(
        self,
        sampling_ratio: float,
        bcs: bool,
        multiplier: float,
        shift: float,
        tfms=None,
        train=False,
    ):
        self.data = torchvision.datasets.MNIST(
            "../input",
            train=train,
            transform=tfms,
            download=True,
        )
        self.bcs = bcs
        self.multiplier = multiplier
        self.shift = shift

        if bcs:
            phi = np.load("../input/phi_block_binary.npy")
            phi = phi[: int(sampling_ratio * 16)]
            phi = torch.FloatTensor(phi)

            self.cs_operator = BCSOperator(phi)
        else:
            pass

    def __getitem__(self, idx):
        image, _ = self.data[idx]
        image_ = image * self.multiplier + self.shift
        image_ = image_.unsqueeze(dim=1)
        y = self.cs_operator(image_)
        return y.squeeze(dim=0), image

    def __len__(self):
        return len(self.data)
