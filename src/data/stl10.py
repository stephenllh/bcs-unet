import os
import cv2
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
from torchvision import transforms
import pytorch_lightning as pl
from .base_dataset import BaseDataset
from utils import create_patches


class STL10Dataset(BaseDataset):
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


class STL10ReconnetTestDataset(BaseDataset):
    """
    The test dataset of ReconNet is unique because we need to reconstruct image patches and combine.
    This is because ReconNet can only be used on small images.
    To reproduce this, crop the 96x96 STL10 images into 32x32.
    """

    def __init__(self, sampling_ratio: float, bcs: bool):
        super().__init__(sampling_ratio=sampling_ratio, bcs=bcs)
        self.root_dir = "../input/STL10"
        self.data_dir = os.path.join(self.root_dir, "test_images_32x32")
        if not os.path.exists(self.data_dir):
            create_patches(self.root_dir, size=32)
        self.filenames = os.listdir(self.data_dir)

    def __getitem__(self, idx):
        image = cv2.imread(
            os.path.join(self.data_dir, self.filenames[idx]),
            cv2.IMREAD_GRAYSCALE,
        )
        image = torch.tensor(image) / 255.0
        image = image.unsqueeze(dim=0)
        image_ = image.unsqueeze(dim=1)
        y = self.cs_operator(image_)
        return y.squeeze(dim=0), image

    def __len__(self):
        return len(self.data)


class STL10DataModule(pl.LightningDataModule):
    """Pytorch Lightning Data Module for PyTorch datasets, e.g. MNIST, FMNIST, SVHN, etc."""

    def __init__(self, config, reconnet=False):
        super().__init__()
        self.config = config
        self.dm_config = config["data_module"]
        self.reconnet = reconnet  # whether or not ReconNet is the architecture.

    def setup(self, stage=None):
        train_tfms_list = [transforms.RandomCrop(32)] if self.reconnet else []
        train_tfms_list += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ]
        train_tfms = transforms.Compose(train_tfms_list)

        val_tfms_list = [transforms.CenterCrop(32)] if self.reconnet else []
        val_tfms_list += [
            transforms.Grayscale(),
            transforms.ToTensor(),
        ]
        val_tfms = transforms.Compose(val_tfms_list)

        self.train_dataset = STL10Dataset(
            sampling_ratio=self.config["sampling_ratio"],
            bcs=self.config["bcs"],
            tfms=train_tfms,
            train=True,
        )

        self.val_dataset = STL10Dataset(
            sampling_ratio=self.config["sampling_ratio"],
            bcs=self.config["bcs"],
            tfms=val_tfms,
            train=True,
        )

        if self.reconnet:
            self.test_dataset = STL10ReconnetTestDataset(
                sampling_ratio=self.config["sampling_ratio"],
                bcs=self.config["bcs"],
            )
        else:
            self.test_dataset = STL10Dataset(
                sampling_ratio=self.config["sampling_ratio"],
                bcs=self.config["bcs"],
                tfms=val_tfms,
                train=False,
            )

        dataset_size = len(self.train_dataset)
        indices = torch.randperm(dataset_size)
        split = int(self.dm_config["val_percent"] * dataset_size)
        self.train_idx, self.valid_idx = indices[split:], indices[:split]

    def train_dataloader(self):
        train_sampler = SubsetRandomSampler(self.train_idx)
        return DataLoader(
            self.train_dataset,
            batch_size=self.dm_config["batch_size"],
            sampler=train_sampler,
            num_workers=self.dm_config["num_workers"],
        )

    def val_dataloader(self):
        val_sampler = SubsetRandomSampler(self.valid_idx)
        return DataLoader(
            self.val_dataset,
            batch_size=self.dm_config["batch_size"],
            sampler=val_sampler,
            num_workers=self.dm_config["num_workers"],
        )

    def test_dataloader(self):
        if self.reconnet:
            batch_size = 9  # 9 image patches per 96x96 image
        else:
            batch_size = self.dm_config["batch_size"]

        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=self.dm_config["num_workers"],
        )

    def predict_dataloader(self):
        if self.reconnet:
            batch_size = 9  # 9 image patches per 96x96 image
        else:
            batch_size = self.dm_config["batch_size"]

        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=self.dm_config["num_workers"],
        )
