import cv2
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
import albumentations as alb
from albumentations.pytorch.transforms import ToTensorV2
import pytorch_lightning as pl
from .base_dataset import BCSOperator


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
