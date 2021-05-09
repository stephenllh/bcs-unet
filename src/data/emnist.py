import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
from torchvision import transforms
import pytorch_lightning as pl
from .base_dataset import BaseDataset


class EMNISTDataset(BaseDataset):
    def __init__(self, sampling_ratio: float, bcs: bool, tfms=None, train=True):
        super().__init__(sampling_ratio=sampling_ratio, bcs=bcs)
        self.data = torchvision.datasets.EMNIST(
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


class EMNISTDataModule(pl.LightningDataModule):
    """Pytorch Lightning Data Module for PyTorch datasets, e.g. MNIST, FMNIST, SVHN, etc."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dm_config = config["data_module"]

    def setup(self, stage=None):
        train_tfms = transforms.Compose(
            [
                transforms.Resize(32),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.Grayscale(),
                transforms.ToTensor(),
            ]
        )
        val_tfms = transforms.Compose(
            [
                transforms.Resize(32),
                transforms.Grayscale(),
                transforms.ToTensor(),
            ]
        )

        self.train_dataset = EMNISTDataset(
            sampling_ratio=self.config["sampling_ratio"],
            bcs=self.config["bcs"],
            tfms=train_tfms,
            train=True,
        )

        self.val_dataset = EMNISTDataset(
            sampling_ratio=self.config["sampling_ratio"],
            bcs=self.config["bcs"],
            tfms=val_tfms,
            train=True,
        )

        self.test_dataset = EMNISTDataset(
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
        )

    def val_dataloader(self):
        val_sampler = SubsetRandomSampler(self.valid_idx)
        return DataLoader(
            self.val_dataset,
            batch_size=self.dm_config["batch_size"],
            sampler=val_sampler,
        )

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.dm_config["batch_size"])

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.dm_config["batch_size"])
