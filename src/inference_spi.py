import os
from pathlib import Path
import time
import warnings
import numpy as np
import cv2
import scipy.ndimage
import scipy.io
import math
import torch
import pytorch_lightning as pl
from engine.learner import BCSUNetLearner
from utils import voltage2pixel, load_config


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.simplefilter("ignore")


def setup():
    inference_config = load_config("../config/inference_config.yaml")
    ds = inference_config["dataset"]
    sr = inference_config["sampling_ratio"]
    checkpoint_path = f"../logs/BCSUNet_{ds}_{sr}/best/checkpoints/last.ckpt"

    train_config_path = os.path.join(
        Path(checkpoint_path).parent.parent, "hparams.yaml"
    )
    train_config = load_config(train_config_path)

    learner = BCSUNetLearner.load_from_checkpoint(
        checkpoint_path=inference_config["checkpoint_path"], config=train_config
    )

    trainer = pl.Trainer(
        gpus=1 if inference_config["gpu"] else 0,
        logger=False,
        default_root_dir="../",
    )
    return learner, trainer


class RealDataset:
    def __init__(self, inference_config):
        self.real_data = inference_config["real_data"]
        self.phi = np.load(inference_config["measurement_matrix"])
        self.c = int(inference_config["sampling_ratio"] * 16)

    def __getitem__(self, idx):
        real_data = self.real_data[idx]
        path = os.path.join("../inference_input", real_data["filename"])
        y_input = scipy.io.loadmat(path)["y"]

        y_input = y_input[
            np.mod(np.arange(len(y_input)), len(y_input) // 64) < self.c
        ]  # discard extra measurements

        y_input = torch.FloatTensor(y_input).permute(1, 0)
        y_input -= y_input.min()
        y_input /= real_data["max"]

        # Permute is necessary because during sampling, we used "channel-last" format.
        # Hence, we need to permute it to become channel-first to match PyTorch "channel-first" format
        y_input = y_input.view(-1, self.c)
        y_input = y_input.permute(1, 0).contiguous()
        y_input = y_input.view(
            -1, int(math.sqrt(y_input.shape[-1])), int(math.sqrt(y_input.shape[-1]))
        )

        y_input = voltage2pixel(
            y_input, self.phi[: self.c], real_data["min"], real_data["max"]
        )
        return y_input

    def __len__(self):
        return len(self.real_data)


def predict_one():
    # TODO: for standard dataset test set.
    return


def deploy(learner):
    """Real experimental data"""
    inference_config = load_config("../config/inference_config.yaml")
    real_dataset = RealDataset(inference_config)
    for x in real_dataset:
        prediction = learner(x.unsqueeze(0))
        prediction = prediction.squeeze().squeeze().cpu().detach().numpy()

        prediction = scipy.ndimage.zoom(prediction, 8, order=0, mode="nearest")
        cv2.imwrite(f"../temp/{time.time()}.png", prediction * 255)
    print("Finished reconstructing SPI images.")


if __name__ == "__main__":
    learner, trainer = setup()
    deploy(learner)
