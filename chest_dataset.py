from typing import Optional, Callable

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.vision import VisionDataset
from pathlib import Path


class ChestVisionDataset(VisionDataset):
    classes = ["atelectasis", "cardiomegaly", "consolidation", "edema", "effusion", "emphysema", "fibrosis", "hernia",
               "infiltration", "mass", "nodule", "pleural_thickening", "pneumonia", "pneumothorax"]
    splits = set("train", "val", "test")

    def __init__(self, root: str, split: str = "train", transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.split = split.lower()

        if self.split not in self.splits:
            raise RuntimeError("Dataset split not recognized. Must be one of train, val or test")

        self.data, self.targets = self._load_data()

    def _load_data(self):
        npz_file = Path(self.root)
        arrays = np.load(npz_file)
        return arrays[f"{self.split}_images"], arrays[f"{self.split}_labels"]

    def __getitem__(self, item):
        image, target = self.data[item], int(self.targets[item])

        image = Image.fromarray(image)

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self) -> int:
        return len(self.data)