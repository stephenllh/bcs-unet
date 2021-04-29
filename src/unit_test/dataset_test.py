from data.data_module import PyTorchDatasetDataModule
from util import load_config


def dataset_test():
    config = load_config("../config/reconnet_config.yaml")
    dm = PyTorchDatasetDataModule(config=config)
    dm.setup()
    y, image = dm.train_dataset[0]
    print(y.shape, y.min(), y.max())
    print(image.shape, image.min(), image.max())


if __name__ == "__main__":
    dataset_test()
