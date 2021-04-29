import os
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger
from data.data_module import PyTorchDatasetDataModule
from .learner import ReconNetLearner


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def load_config(config_path):
    with open(os.path.join(config_path)) as file:
        config = yaml.safe_load(file)
    return config


def run():
    # main_config = load_config("../config/main_config.yaml")
    # config = load_config(f"../config/{main_config['config_filename']}")
    config = load_config("../config/reconnet_config.yaml")
    data_module = PyTorchDatasetDataModule(config)
    learner = ReconNetLearner(config)
    callbacks = [
        ModelCheckpoint(**config["callbacks"]["checkpoint"]),
        EarlyStopping(**config["callbacks"]["early_stopping"]),
        LearningRateMonitor(),
    ]

    dataset_name = config['dataset_name']
    sampling_ratio = config['sampling_ratio']
    log_name = f"reconnet_{dataset_name}_{int(sampling_ratio * 10000)}"
    logger = TensorBoardLogger(save_dir="../logs", name=log_name)

    message = f"Running ReconNet on {dataset_name} dataset. Sampling ratio = {sampling_ratio * 100}%"
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


if __name__ == "__main__":
    run()
