import os
import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision


class BSDS500Dataset:
    def __init__(
        self, sampling_ratio: float, bcs: bool, train_val_test: str, tfms=None
    ):
        self.bcs = bcs  # TODO: add the bcs and ccs functionality
        self.path = "../input/BSDS500_crop32/" + train_val_test
        self.filenames = [filename for filename in os.listdir(self.path)]
        self.tfms = tfms
        if bcs:
            phi = np.load("../input/phi_block_binary.npy")
            phi = phi[: int(sampling_ratio * 16)]
            phi = torch.FloatTensor(phi)
            self.cs_operator = BCSOperator(phi)
        else:  # TODO: prepare for CCS
            pass
            # phi = np.load("../input/________.npy")  # TODO: make the CCS phi format correct
            # self.cs_operator = CCSOperator(phi)

    def __getitem__(self, idx):
        image = cv2.imread(os.path.join(self.path, self.filenames[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.tfms is not None:
            augmented = self.tfms(image=image)
            image = augmented["image"]
        image_ = image.unsqueeze(dim=1)
        y = self.cs_operator(image_)

        return y.squeeze(dim=0), image

    def __len__(self):
        return len(self.filenames)


class PyTorchBaseDataset:
    def __init__(self, sampling_ratio: float, bcs: bool):
        if bcs:
            phi = np.load(
                "../input/phi_block_binary.npy"
            )  # TODO: make the hardcoded path a variable.
            phi = phi[: int(sampling_ratio * 16)]
            phi = torch.FloatTensor(phi)
            # print("phi_shape", phi.shape)
            self.cs_operator = BCSOperator(phi)
        else:  # TODO: prepare for CCS
            pass
            # phi = np.load("../input/________.npy")  # TODO: make the CCS phi format correct
            # self.cs_operator = CCSOperator(phi)


class STL10Dataset(PyTorchBaseDataset):
    def __init__(self, sampling_ratio: float, bcs: bool, tfms=None, train=True):
        super().__init__(sampling_ratio=sampling_ratio, bcs=bcs)
        self.data = torchvision.datasets.STL10(
            "../input/STL10",
            split="unlabeled" if train else "test",
            transform=tfms,
            download=True,
        )

    def __getitem__(self, idx):
        image, _ = self.data[idx]
        image_ = image.unsqueeze(dim=1)
        y = self.cs_operator(image_)
        return y.squeeze(dim=0), image

    def __len__(self):
        return len(self.data)


class SVHNDataset(PyTorchBaseDataset):
    def __init__(self, sampling_ratio: float, bcs: bool, tfms=None, train=True):
        super().__init__(sampling_ratio=sampling_ratio, bcs=bcs)
        self.data = torchvision.datasets.SVHN(
            "../input/SVHN",
            split="train" if train else "test",
            transform=tfms,
            download=True,
        )

    def __getitem__(self, idx):
        image, _ = self.data[idx]
        image_ = image.unsqueeze(dim=1)
        y = self.cs_operator(image_)
        return y.squeeze(dim=0), image

    def __len__(self):
        return len(self.data)


class MNISTDataset(PyTorchBaseDataset):
    def __init__(self, sampling_ratio: float, bcs: bool, tfms=None, train=True):
        super().__init__(sampling_ratio=sampling_ratio, bcs=bcs)
        self.data = torchvision.datasets.MNIST(
            "../input",
            train=train,
            transform=tfms,
            download=True,
        )

    def __getitem__(self, idx):
        image, _ = self.data[idx]
        image_ = image.unsqueeze(dim=1)
        y = self.cs_operator(image_)
        return y.squeeze(dim=0), image

    def __len__(self):
        return len(self.data)


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
