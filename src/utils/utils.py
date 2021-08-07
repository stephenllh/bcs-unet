import os
import yaml
import torch
import math
import cv2


def voltage2pixel(y, phi, low, high):
    """
    Converts the measurement tensor / vector from voltage scale to pixel scale,
    so that the neural network can understand.
    y - measurement
    phi - block measurement matrix
    low - lowest voltage pixel
    high - highest voltage pixel
    """

    if type(y) == "numpy.ndarray":
        y = torch.from_numpy(y).float()

    phi = torch.from_numpy(phi).float()

    if y.dim() == 4 and y.shape[0] == 1:
        y = y.squeeze(dim=0)

    if phi.dim() == 4 and phi.shape[1] == 1:
        phi = phi.squeeze(dim=1)

    term1 = y / (high - low)
    term2 = (phi.sum(dim=(1, 2)) * low / (high - low)).unsqueeze(-1).unsqueeze(-1)

    y_pixel_scale = term1 - term2
    return y_pixel_scale


def load_config(config_path):
    with open(os.path.join(config_path)) as file:
        config = yaml.safe_load(file)
    return config


def reshape_into_block(y, sr: float, block_size=4):
    c = int(sr * block_size ** 2)
    h = w = int(math.sqrt(y.shape[0] // c))
    y = y.reshape(h, w, c)
    y = y.transpose((2, 0, 1))
    return y


def create_patches(root_dir, size):
    print(f"Creating {size}x{size} image patches for STL10 test set.")
    dataset_dir = os.path.join(root_dir, "test_images")
    save_dir = os.path.join(root_dir, "test_images_32x32")

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)

    filenames = os.listdir(dataset_dir)
    for filename in filenames:
        img = cv2.imread(f"{dataset_dir}/{filename}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.reshape(img.shape[0] // size, size, img.shape[1] // size, size)
        img = img.transpose(0, 2, 1, 3).reshape(-1, size, size)
        for idx, img_patch in enumerate(img):
            save_filename = f"{save_dir}/{filename.split('.')[0]}_{idx}.png"
            cv2.imwrite(save_filename, img_patch)
