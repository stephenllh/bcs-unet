import os
import argparse
import numpy as np
from pathlib import Path
import warnings
import cv2
import scipy.ndimage
from data.emnist import EMNISTDataModule
from data.svhn import SVHNDataModule
from data.stl10 import STL10DataModule
from .learner import ReconNetLearner
from utils import load_config, create_patches


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

    checkpoint_folder = f"../logs/ReconNet_STL10_{int(sr * 100):04d}/best"

    if not os.path.exists(checkpoint_folder):
        run_name = os.listdir(Path(checkpoint_folder).parent)[0]
        checkpoint_path = (
            f"{Path(checkpoint_folder).parent}/{run_name}/checkpoints/last.ckpt"
        )
        message = (
            f"The checkpoint from the run '{run_name}' is selected by default."
            + "If this is not intended, change the name of the preferred checkpoint folder to 'best'."
        )
        print(message)
    else:
        checkpoint_path = f"{checkpoint_folder}/checkpoints/last.ckpt"

    train_config_path = os.path.join(
        Path(checkpoint_path).parent.parent, "hparams.yaml"
    )
    train_config = load_config(train_config_path)
    train_config["sampling_ratio"] = sr / 100
    train_config["img_dim"] = 32

    if dataset == "EMNIST":
        data_module = EMNISTDataModule(train_config)
    elif dataset == "SVHN":
        data_module = SVHNDataModule(train_config)
    elif dataset == "STL10":
        data_module = STL10DataModule(train_config, reconnet=True)

    learner = ReconNetLearner.load_from_checkpoint(
        checkpoint_path=checkpoint_path, config=train_config, strict=False
    )

    message = f"Inference: ReconNet on {dataset} dataset. Sampling ratio = {train_config['sampling_ratio']}"
    print(message)

    data_module.setup()
    ds = data_module.test_dataset

    directory = f"../inference_images/ReconNet/{dataset}/{int(sr * 100):04d}"
    os.makedirs(directory, exist_ok=True)

    if dataset != "STL10":
        for i in np.linspace(0, len(ds) - 1, 30, dtype=int):
            input = ds[i][0].unsqueeze(0)
            out = learner(input).squeeze().squeeze().detach().numpy()
            out = scipy.ndimage.zoom(out, 8, order=0, mode="nearest")
            cv2.imwrite(f"{directory}/{i}.png", out * 255)

    else:
        if not os.path.exists("../input/STL10/test_images_32x32"):
            create_patches("../input/STL10", size=32)
        for i in range(10):
            combined = np.zeros((96, 96), dtype=int)
            for j in range(9):
                input = ds[i * 9 + j][0].unsqueeze(0)
                out = learner(input).squeeze().squeeze().detach().numpy()
                out = (out * 255).astype(int)
                x = j // 3 * 32
                y = j % 3 * 32
                combined[x : x + 32, y : y + 32] = out
            combined = scipy.ndimage.zoom(combined, 8, order=0, mode="nearest")
            cv2.imwrite(f"{directory}/{i}.png", combined)

    print("Done.")


if __name__ == "__main__":
    run()
