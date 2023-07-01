from typing import Optional, Callable

import numpy as np
from PIL import Image
from torchvision.datasets.vision import VisionDataset


class NpzVisionDataset(VisionDataset):
    # classes = ["atelectasis", "cardiomegaly", "consolidation", "edema", "effusion", "emphysema", "fibrosis", "hernia",
    #            "infiltration", "mass", "nodule", "pleural_thickening", "pneumonia", "pneumothorax"]
    classes = ["foo"]
    splits = ["train", "val", "test"]

    def __init__(self, root: str, split: str = "train", to_rgb: bool = False, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.split = split.lower()
        self.to_rgb = to_rgb

        if self.split not in self.splits:
            raise RuntimeError("Dataset split not recognized. Must be one of "
                               + ",".join(split[:-1]) + f" or {split[-1]}")

        self.images, self.labels = self._load_dataset()

    def _load_dataset(self):
        arrays = np.load(self.root)
        return arrays[f"{self.split}_images"], arrays[f"{self.split}_labels"]

    def __getitem__(self, item):
        image, target = self.images[item], self.labels[item]

        if self.to_rgb or self.transform:
            image = Image.fromarray(image)

            if self.to_rgb:
                image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self) -> int:
        return len(self.images)
