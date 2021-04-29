import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
import pytorch_lightning as pl
from data.dataset import MNISTDataset, SVHNDataset  # , BSDS500Dataset


class BSDS500DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        csv_path,
        batch_size,
        fold=0,
        train_transforms=None,
        test_transforms=None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.dataframe = pd.read_csv(csv_path)
        self.batch_size = batch_size
        self.fold = fold
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms

    def setup(self, stage=None):
        # TODO
        # self.train_dataset = BSDS500Dataset(
        #     dataframe=self.dataframe,
        #     data_dir=self.data_dir,
        #     mode="train",
        #     tfms=self.train_transforms,
        # )

        # self.valid_dataset = BSDS500Dataset(
        #     dataframe=self.dataframe,
        #     data_dir=self.data_dir,
        #     mode="valid",
        #     tfms=self.test_transforms,
        # )
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


class PyTorchDatasetDataModule(pl.LightningDataModule):
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
                # transforms.Normalize((0.0,), (1.0,)),
            ]
        )
        val_tfms = transforms.Compose(
            [
                transforms.Resize(32),
                transforms.Grayscale(),
                transforms.ToTensor(),
                # transforms.Normalize((0.0,), (1.0,)),
            ]
        )
        dataset_name = self.config["dataset_name"]
        if dataset_name == "mnist":
            Dataset = MNISTDataset
        # elif dataset_name == "fmnist":
        #     Dataset = FMNISTDataset
        elif dataset_name == "svhn":
            Dataset = SVHNDataset

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
        torch.manual_seed(0)
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
