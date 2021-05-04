import os
from pathlib import Path
import yaml
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import torch

# from torch.utils.data import DataLoader
import pytorch_lightning as pl
from data.data_module import PyTorchDatasetDataModule
from .learner import ReconNetLearner


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# TODO: merge (or refactor) to make it the main inference script.


def load_config(config_path):
    with open(os.path.join(config_path)) as file:
        config = yaml.safe_load(file)
    return config


def setup():
    inference_config = load_config("../config/inference_config.yaml")
    checkpoint_path = inference_config["checkpoint_path"]
    train_config_path = os.path.join(
        Path(checkpoint_path).parent.parent, "hparams.yaml"
    )
    train_config = load_config(train_config_path)

    data_module = PyTorchDatasetDataModule(train_config)
    learner = ReconNetLearner.load_from_checkpoint(
        checkpoint_path=inference_config["checkpoint_path"], config=train_config
    )
    trainer = pl.Trainer(
        gpus=inference_config["gpu"],
        logger=False,
        default_root_dir="../",
        progress_bar_refresh_rate=20,
        limit_test_batches=(2 if inference_config["test_mode"] else 1.0),
    )
    return data_module, learner, trainer


def get_test_metrics(data_module, learner, trainer):
    test_metrics = trainer.test(learner, datamodule=data_module)
    return test_metrics


class RealDataset:
    def __init__(self, inference_config):
        self.real_data = inference_config["real_data"]
        self.phi = np.load(inference_config["measurement_matrix"])
        self.c = int(inference_config["sampling_ratio"] * 16)

    def __getitem__(self, idx):
        real_data = self.real_data[idx]
        path = os.path.join("../inference_input", real_data["filename"])
        y_input = scipy.io.loadmat(path)["y"]

        y_input = torch.FloatTensor(y_input).permute(1, 0)
        y_input -= y_input.min()
        y_input /= real_data["max"]

        # Permute is necessary because during sampling, we used "channel-last" format.
        # Hence, we need to permute it to become channel-first to match PyTorch "channel-first" format
        y_input = y_input.view(-1, self.c)
        y_input = y_input.permute(1, 0).contiguous()
        y_input = y_input.view(1, -1)
        return y_input

    def __len__(self):
        return len(self.real_data)


def predict_one():
    # TODO: decide if this function is for real data or standard dataset test set.
    return


def deploy(learner):
    """Real experimental data"""
    inference_config = load_config("../config/inference_config.yaml")
    real_dataset = RealDataset(inference_config)
    # print("rdshape", real_dataset[0].shape)
    # print(real_dataset[0].max(), real_dataset[0].min())
    # inference_loader = DataLoader(real_dataset, batch_size=32)
    predictions = learner(real_dataset[0].unsqueeze(0))
    plt.imshow(predictions.squeeze().squeeze().cpu().detach().numpy(), cmap="gray")
    plt.axis("off")
    plt.show()
    # return predictions


if __name__ == "__main__":
    data_module, learner, trainer = setup()
    # get_test_metrics(data_module, learner, trainer)
    # predict_test_set(data_module, learner, trainer)
    deploy(learner)
