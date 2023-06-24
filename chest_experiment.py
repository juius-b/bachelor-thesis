import argparse
import os
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader

import numpy as np
import torch


def main():
    npz_file = Path("/mnt/jbrockma/bachelor-thesis-npz/chest.npz")
    chest_arrays = np.load(npz_file)

    cuda_device_id = os.environ["CUDA_VISIBLE_DEVICES"]

    device = torch.device(cuda_device_id) if torch.cuda.is_available() else torch.device("cpu")

    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])


if __name__ == '__main__':
    main()