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
from .learner import ReconNetLearner
from utils import load_config

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

parser = argparse.ArgumentParser(description="Wheat detection with EfficientDet")
parser.add_argument("-d", "--dataset", type=str, help="'EMNIST', 'SVHN', or 'STL10'")
args = parser.parse_args()


def run():
    seed_everything(seed=0, workers=True)

    config = load_config("../config/reconnet_config.yaml")

    if args.dataset == "EMNIST":
        data_module = EMNISTDataModule(config)
    elif args.dataset == "SVHN":
        data_module = SVHNDataModule(config)
    elif args.dataset == "STL10":
        config["data_module"]["num_workers"] = 0
        data_module = STL10DataModule(config, reconnet=True)
    else:
        raise NotImplementedError

    learner = ReconNetLearner(config)
    callbacks = [
        ModelCheckpoint(**config["callbacks"]["checkpoint"]),
        EarlyStopping(**config["callbacks"]["early_stopping"]),
        LearningRateMonitor(),
    ]

    sampling_ratio = config["sampling_ratio"]
    log_name = f"ReconNet_{args.dataset}_{int(sampling_ratio * 10000)}"
    logger = TensorBoardLogger(save_dir="../logs", name=log_name)

    message = f"Running ReconNet on {args.dataset} dataset. Sampling ratio = {sampling_ratio * 100}%"
    print("-" * 100)
    print(message)
    print("-" * 100)

    trainer = pl.Trainer(
        gpus=config["trainer"]["gpu"],
        max_epochs=config["trainer"]["epochs"],
        default_root_dir="../",
        progress_bar_refresh_rate=20,
        callbacks=callbacks,
        precision=(16 if config["trainer"]["fp16"] else 32),
        logger=logger,
    )
    trainer.fit(learner, data_module)
    trainer.test(learner, datamodule=data_module, ckpt_path="best")


if __name__ == "__main__":
    run()
