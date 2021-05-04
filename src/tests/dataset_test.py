from data.data_module import PyTorchDatasetDataModule, BSDS500DataModule
from util import load_config


def pytorch_dataset_test():
    config = load_config("../config/reconnet_config.yaml")
    dm = PyTorchDatasetDataModule(config=config)
    dm.setup()
    y, image = dm.train_dataset[0]
    print(y.shape, y.min(), y.max())
    print(image.shape, image.min(), image.max())


def BSDS500_test():
    config = load_config("../config/reconnet_config.yaml")
    dm = BSDS500DataModule(config=config)
    dm.setup()
    y, image = dm.train_dataset[0]
    print(y.shape, y.min(), y.max())
    print(image.shape, image.min(), image.max())


if __name__ == "__main__":
    pytorch_dataset_test()
    # BSDS500_test()
