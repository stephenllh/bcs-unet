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
parser.add_argument("-d", "--dataset", type=str, help="'emnist', 'svhn', or 'stl10'")
args = parser.parse_args()


def run():
    seed_everything(seed=0, workers=True)

    config = load_config("../config/bcsunet_config.yaml")

    if args.dataset == "EMNIST":
        data_module = EMNISTDataModule(config)
    elif args.dataset == "SVHN":
        data_module = SVHNDataModule(config)
    elif args.dataset == "STL10":
        data_module = STL10DataModule(config)

    learner = SCSNetLearner(config)
    callbacks = [
        ModelCheckpoint(**config["callbacks"]["checkpoint"]),
        EarlyStopping(**config["callbacks"]["early_stopping"]),
        LearningRateMonitor(),
    ]

    log_name = (
        f"BCSUNet_{config['dataset_name']}_{int(config['sampling_ratio'] * 10000)}"
    )
    logger = TensorBoardLogger(save_dir="../logs", name=log_name)

    message = f"Running SCSNet on {config['dataset_name']} dataset. Sampling ratio = {config['sampling_ratio']}"
    print("-" * 100)
    print(message)
    print("-" * 100)

    trainer = pl.Trainer(
        gpus=config["trainer"]["gpu"],
        max_epochs=1,
        # max_epochs=config["trainer"]["epochs"],
        default_root_dir="../",
        callbacks=callbacks,
        precision=(16 if config["trainer"]["fp16"] else 32),
        logger=logger,
    )
    trainer.fit(learner, data_module)
    trainer.test(learner, data_module, ckpt_path="best")


if __name__ == "__main__":
    run()
