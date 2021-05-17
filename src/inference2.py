import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from data.emnist import EMNISTDataModule
from data.svhn import SVHNDataModule
from data.stl10 import STL10DataModule
from .learner import SCSNetLearner
from utils import load_config


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

parser = argparse.ArgumentParser(description="Wheat detection with EfficientDet")
parser.add_argument("-d", "--dataset", type=str, help="'EMNIST', 'SVHN', or 'STL10'")
args = parser.parse_args()


def run():
    seed_everything(seed=0, workers=True)

    config = load_config("../config/scsnet_config.yaml")

    if args.dataset == "EMNIST":
        data_module = EMNISTDataModule(config)
    elif args.dataset == "SVHN":
        data_module = SVHNDataModule(config)
    elif args.dataset == "STL10":
        config["data_module"]["batch_size"] = 64
        data_module = STL10DataModule(config)
    else:
        raise NotImplementedError

    PATH = "../logs/SCSNet_STL10_1250/version_0/checkpoints/epoch=22-step=32360.ckpt"
    learner = SCSNetLearner.load_from_checkpoint(PATH, config)

    message = f"Running SCSNet on {args.dataset} dataset. Sampling ratio = {config['sampling_ratio']}"
    print("-" * 100)
    print(message)
    print("-" * 100)

    trainer = pl.Trainer(
        gpus=config["trainer"]["gpu"],
        max_epochs=1,
        default_root_dir="../",
        precision=(16 if config["trainer"]["fp16"] else 32),
        logger=None,
    )
    trainer.test(learner, datamodule=data_module, ckpt_path="best")


if __name__ == "__main__":
    run()
