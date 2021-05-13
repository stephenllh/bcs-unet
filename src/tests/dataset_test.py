from data.stl10 import STL10DataModule
from utils import load_config


# def pytorch_dataset_test():
#     config = load_config("../config/reconnet_config.yaml")
#     dm = PyTorchDatasetDataModule(config=config)
#     dm.setup()
#     y, image = dm.train_dataset[0]
#     print(y.shape, y.min(), y.max())
#     print(image.shape, image.min(), image.max())


def stl10_test_dataset_test():
    config = load_config("../config/reconnet_config.yaml")
    dm = STL10DataModule(config, reconnet=True)
    dm.setup()
    y, image = dm.test_dataset[0]
    print(y.shape, y.min(), y.max())
    print(image.shape, image.min(), image.max())

# def BSDS500_test():
#     config = load_config("../config/reconnet_config.yaml")
#     dm = BSDS500DataModule(config=config)
#     dm.setup()
#     y, image = dm.train_dataset[0]
#     print(y.shape, y.min(), y.max())
#     print(image.shape, image.min(), image.max())


if __name__ == "__main__":
    stl10_test_dataset_test()

