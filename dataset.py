from typing import Optional, Callable, List, Type, Dict, Any

import numpy as np
from PIL import Image
from torchvision.datasets.vision import VisionDataset


class NpzVisionDataset(VisionDataset):
    splits = ["train", "val", "test"]
    classes: List[str]

    def __init__(self, root: str, split: str = "train", transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.split = split.lower()

        if self.split not in self.splits:
            raise RuntimeError("Dataset split not recognized. Must be one of "
                               + ",".join(split[:-1]) + f" or {split[-1]}")

        self.data, self.targets = self._load_dataset()

    def _load_dataset(self):
        arrays = np.load(self.root)
        return arrays[f"{self.split}_images"], arrays[f"{self.split}_labels"]

    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = Image.fromarray(img)
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)


BUILTIN_DATASETS = {}


def register_dataset(name: str = None):
    def wrapper(cls: Type[NpzVisionDataset]) -> Type[NpzVisionDataset]:
        BUILTIN_DATASETS[name] = cls
        return cls

    return wrapper


@register_dataset("chest")
class ChestDataset(NpzVisionDataset):
    classes = ["atelectasis", "cardiomegaly", "consolidation", "edema", "effusion", "emphysema", "fibrosis", "hernia",
               "infiltration", "mass", "nodule", "pleural_thickening", "pneumonia", "pneumothorax"]


@register_dataset("breast")
class BreastDataset(NpzVisionDataset):
    classes = ["class"]


def create_dataset_builder(name: str, *args, **kwargs):
    def builder(*_args, **_kwargs):
        all_args = args + _args
        all_kwargs = {**kwargs, **_kwargs}
        return get_dataset_class(name)(*all_args, **all_kwargs)

    return builder


def get_dataset_class(name: str) -> Type[NpzVisionDataset]:
    name = name.lower()
    try:
        cls = BUILTIN_DATASETS[name]
    except KeyError:
        raise ValueError(f"Unknown dataset {name}")
    return cls


def get_dataset(name: str, **config: Any) -> NpzVisionDataset:
    cls = get_dataset_class(name)
    return cls(**config)
