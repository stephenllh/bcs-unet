import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
import albumentations as alb
from albumentations.pytorch.transforms import ToTensorV2
import pytorch_lightning as pl
from data.dataset import MNISTDataset, SVHNDataset, BSDS500Dataset, STL10Dataset


class BSDS500DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dm_config = config["data_module"]

    def setup(self, stage=None):
        train_tfms = alb.Compose(
            [
                # alb.CenterCrop(256, 256),
                alb.HorizontalFlip(p=0.5),
                alb.ColorJitter(brightness=0.2, contrast=0.2),
                alb.Normalize(mean=(0,), std=(1,)),
                ToTensorV2(),
            ]
        )
        val_tfms = alb.Compose(
            [
                # alb.CenterCrop(256, 256),
                alb.Normalize(mean=(0,), std=(1,)),
                ToTensorV2(),
            ]
        )

        self.train_dataset = BSDS500Dataset(
            sampling_ratio=self.dm_config["sampling_ratio"],
            bcs=self.config["bcs"],
            train_val_test="train",
            tfms=train_tfms,
        )

        self.val_dataset = BSDS500Dataset(
            sampling_ratio=self.dm_config["sampling_ratio"],
            bcs=self.config["bcs"],
            train_val_test="val",
            tfms=val_tfms,
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.dm_config["batch_size"])

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.dm_config["batch_size"])

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.dm_config["batch_size"])


class PyTorchDatasetDataModule(pl.LightningDataModule):
    """Pytorch Lightning Data Module for PyTorch datasets, e.g. MNIST, FMNIST, SVHN, etc."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dm_config = config["data_module"]

    def setup(self, stage=None):
        dataset_name = self.config["dataset_name"]
        if dataset_name in ["mnist", "svhn"]:
            image_size = 32
        else:
            image_size = 96

        train_tfms = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.Grayscale(),
                transforms.ToTensor(),
            ]
        )
        val_tfms = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.Grayscale(),
                transforms.ToTensor(),
            ]
        )

        if dataset_name == "mnist":
            Dataset = MNISTDataset
        elif dataset_name == "svhn":
            Dataset = SVHNDataset
        elif dataset_name == "stl10":
            Dataset = STL10Dataset

        self.train_dataset = Dataset(
            sampling_ratio=self.config["sampling_ratio"],
            bcs=self.config["bcs"],
            tfms=train_tfms,
            train=True,
        )

        self.val_dataset = Dataset(
            sampling_ratio=self.config["sampling_ratio"],
            bcs=self.config["bcs"],
            tfms=val_tfms,
            train=True,
        )

        self.test_dataset = Dataset(
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
