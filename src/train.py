import os
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from data.data_module import ImagenetteDataModule
from data.transforms import get_transforms
from model.nets import SimpleNet
from engine.learner import ImagenetteClassifier


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def load_config(config_path):
    with open(os.path.join(config_path)) as file:
        config = yaml.safe_load(file)
    return config


def run():
    config = load_config("../config/config.yaml")

    # TODO: transforms
    data = ImagenetteDataModule(
        **config["data_module"],
        train_transforms=get_transforms(config["transforms"], is_train=True),
        test_transforms=get_transforms(config["transforms"], is_train=False),
    )

    # TODO: configure net with config
    simplenet = SimpleNet()
    learner = ImagenetteClassifier(simplenet, config)

    callbacks = [
        ModelCheckpoint(**config["callbacks"]["checkpoint"]),
        EarlyStopping(**config["callbacks"]["early_stopping"]),
        LearningRateMonitor(),
    ]

    # Initialize a trainer
    trainer = pl.Trainer(
        gpus=config["trainer"]["gpu"],
        max_epochs=config["trainer"]["epochs"],
        default_root_dir="../",
        progress_bar_refresh_rate=20,
        callbacks=callbacks,
    )
    trainer.fit(learner, data)


if __name__ == "__main__":
    run()
