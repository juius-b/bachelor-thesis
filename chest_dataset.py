from typing import Optional, Callable

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.vision import VisionDataset
from pathlib import Path


class ChestVisionDataset(VisionDataset):
    classes = ["atelectasis", "cardiomegaly", "consolidation", "edema", "effusion", "emphysema", "fibrosis", "hernia",
               "infiltration", "mass", "nodule", "pleural_thickening", "pneumonia", "pneumothorax"]
    splits = {"train", "val", "test"}

    def __init__(self, root: str, split: str = "train", to_rgb: bool = False, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.split = split.lower()
        self.to_rgb = to_rgb

        if self.split not in self.splits:
            raise RuntimeError("Dataset split not recognized. Must be one of train, val or test")

        self.images, self.labels = self._load_dataset()

    def _load_dataset(self):
        npz_file = Path(self.root)
        arrays = np.load(npz_file)
        return arrays[f"{self.split}_images"], arrays[f"{self.split}_labels"]

    def __getitem__(self, item):
        image, label = self.images[item], self.labels[item]

        image = Image.fromarray(image)

        if self.to_rgb:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self) -> int:
        return len(self.images)
