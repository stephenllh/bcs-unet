import torch
from torch.utils.data import DataLoader
from data.data_module import ImagewoofDataset

import pandas as pd
from data.transforms import get_transforms
from model.nets import SimpleNet
from engine.learner import STL10Classifier


def run():
    print("Start.")
    DATA_DIR = "C:/DL/Datasets/imagewoof"
    dataframe = pd.read_csv(f"{DATA_DIR}/noisy_imagewoof.csv")
    test_transforms = get_transforms(image_size=128, is_train=False)
    test_dataset = ImagewoofDataset(
        dataframe=dataframe,
        data_dir=DATA_DIR,
        mode="test",
        tfms=test_transforms,
    )
    test_loader = DataLoader(test_dataset, batch_size=32)

    ckpt_path = "checkpoint/last.ckpt"
    simplenet = SimpleNet().cuda()
    model = STL10Classifier.load_from_checkpoint(
        checkpoint_path=ckpt_path, net=simplenet
    )
    # checkpoint_callback = ModelCheckpoint(save_last=True, verbose=True, mode="min")

    # Obtain predictions
    predictions = []
    filenames = []
    for image_batch, filename_batch in test_loader:
        # print(filename_batch)
        pred_batch = model(image_batch.cuda())
        pred_batch = torch.argmax(pred_batch, dim=1)
        predictions.extend(pred_batch.cpu().detach().numpy().tolist())
        filenames.extend(filename_batch)

    submission = pd.DataFrame(columns=["filename", "label"])
    submission["filename"] = filenames
    submission["label"] = predictions
    submission.to_csv("submission.csv", index=False)
    print("Done.")


if __name__ == "__main__":
    run()
