import os
import numpy as np
import argparse
from pathlib import Path
import warnings
import cv2
import scipy.ndimage
from data.emnist import EMNISTDataModule
from data.svhn import SVHNDataModule
from data.stl10 import STL10DataModule
from engine.learner import BCSUNetLearner
from utils import load_config


parser = argparse.ArgumentParser()
parser.add_argument(
    "-d", "--dataset", type=str, required=True, help="'EMNIST', 'SVHN', or 'STL10'"
)
parser.add_argument(
    "-s",
    "--sampling_ratio",
    type=float,
    required=True,
    help="Sampling ratio in percentage",
)
args = parser.parse_args()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.simplefilter("ignore")


def run():
    dataset = args.dataset
    sr = args.sampling_ratio
    checkpoint_path = (
        f"../logs/BCSUNet_{dataset}_{int(sr * 100):04d}/best/checkpoints/last.ckpt"
    )
    train_config_path = os.path.join(
        Path(checkpoint_path).parent.parent, "hparams.yaml"
    )
    train_config = load_config(train_config_path)
    train_config["sampling_ratio"] = sr / 100

    if dataset == "EMNIST":
        data_module = EMNISTDataModule(train_config)
    elif dataset == "SVHN":
        data_module = SVHNDataModule(train_config)
    elif dataset == "STL10":
        data_module = STL10DataModule(train_config)

    learner = BCSUNetLearner.load_from_checkpoint(
        checkpoint_path=checkpoint_path, config=train_config, strict=False
    )

    message = f"Inference: BCS-UNet on {dataset} dataset. Sampling ratio = {train_config['sampling_ratio']}"
    print(message)

    data_module.setup()
    ds = data_module.test_dataset

    directory = f"../inference_images/BCSUNet/{dataset}/{int(sr * 100):04d}"
    os.makedirs(directory, exist_ok=True)

    for i in np.linspace(0, len(ds) - 1, 30, dtype=int):
        input = ds[i][0].unsqueeze(0)
        out = learner(input).squeeze().squeeze().detach().numpy()
        out = scipy.ndimage.zoom(out, 8, order=0, mode="nearest")
        cv2.imwrite(f"{directory}/{i}.png", out * 255)

    print("Done.")


if __name__ == "__main__":
    run()
