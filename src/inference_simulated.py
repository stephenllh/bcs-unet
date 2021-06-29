import os
from pathlib import Path
import warnings
import cv2
import scipy.ndimage
from data.emnist import EMNISTDataModule
from data.svhn import SVHNDataModule
from data.stl10 import STL10DataModule
from engine.learner import BCSUNetLearner
from utils import load_config


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.simplefilter("ignore")


def run():
    inference_config = load_config("../config/inference_config.yaml")
    dataset = inference_config["dataset"]
    sr = inference_config["sampling_ratio"]
    checkpoint_path = (
        f"../logs/BCSUNet_{dataset}_{int(sr * 10000):04d}/best/checkpoints/last.ckpt"
    )
    train_config_path = os.path.join(
        Path(checkpoint_path).parent.parent, "hparams.yaml"
    )
    train_config = load_config(train_config_path)
    train_config["sampling_ratio"] = sr

    if dataset == "EMNIST":
        data_module = EMNISTDataModule(train_config)
    elif dataset == "SVHN":
        data_module = SVHNDataModule(train_config)
    elif dataset == "STL10":
        data_module = STL10DataModule(train_config)

    learner = BCSUNetLearner.load_from_checkpoint(
        checkpoint_path=checkpoint_path, config=train_config
    )

    message = f"Inference: BCS-UNet on {dataset} dataset. Sampling ratio = {train_config['sampling_ratio']}"
    print(message)

    data_module.setup()
    ds = data_module.test_dataset

    directory = f"../temp/{dataset}/{int(sr * 10000):04d}"
    if not os.path.exists(directory):
        os.makedirs(directory)

    for i in range(30):
        input = ds[i][0].unsqueeze(0)
        out = learner(input).squeeze().squeeze().detach().numpy()
        out = scipy.ndimage.zoom(out, 8, order=0, mode="nearest")
        cv2.imwrite(f"{directory}/{i}.png", out * 255)

    print("Done.")
    # trainer = pl.Trainer(
    #     gpus=train_config["trainer"]["gpu"],
    #     default_root_dir="../",
    #     precision=(16 if train_config["trainer"]["fp16"] else 32),
    # )
    # trainer.predict(learner, datamodule=data_module, ckpt_path="best")


if __name__ == "__main__":
    run()
