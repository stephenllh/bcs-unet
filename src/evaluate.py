import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything
from data.emnist import EMNISTDataModule
from data.svhn import SVHNDataModule
from data.stl10 import STL10DataModule
from engine.learner import BCSUNetLearner
from utils import load_config


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

parser = argparse.ArgumentParser(description="Wheat detection with EfficientDet")
parser.add_argument("-d", "--dataset", type=str, help="'EMNIST', 'SVHN', or 'STL10'")
parser.add_argument(
    "-s",
    "--sampling_ratio",
    type=float,
    required=True,
    help="Sampling ratio in percentage",
)

args = parser.parse_args()


def run():
    seed_everything(seed=0, workers=True)
    path = f"../logs/BCSUNet_{args.dataset}_{int(args.sampling_ratio * 10000)}"
    config = load_config(f"{path}/version_0/hparams.yaml")

    if args.dataset == "EMNIST":
        data_module = EMNISTDataModule(config)
    elif args.dataset == "SVHN":
        data_module = SVHNDataModule(config)
    elif args.dataset == "STL10":
        data_module = STL10DataModule(config)
    else:
        raise NotImplementedError

    PATH = f"{path}/version_0/checkpoints/best.ckpt"
    learner = BCSUNetLearner.load_from_checkpoint(PATH, config)

    trainer = pl.Trainer(
        gpus=1,
        default_root_dir="../",
        logger=False,
    )
    trainer.test(learner, datamodule=data_module)


if __name__ == "__main__":
    run()
