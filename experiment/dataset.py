import medmnist.info
import numpy as np
from PIL import Image
from torchvision.datasets.vision import VisionDataset
from typing import Optional, Callable, List, Type, Any


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


def register_dataset(name: str, problem: str):
    def wrapper(cls: Type[NpzVisionDataset]) -> Type[NpzVisionDataset]:
        BUILTIN_DATASETS[name] = (cls, problem)
        return cls

    return wrapper


@register_dataset("path", "multiclass")
class PathDataset(NpzVisionDataset):
    classes = medmnist.info.INFO["pathmnist"]["label"].values()


@register_dataset("chest", "multilabel")
class ChestDataset(NpzVisionDataset):
    classes = medmnist.info.INFO["chestmnist"]["label"].values()


@register_dataset("derma", "multiclass")
class DermaDataset(NpzVisionDataset):
    classes = medmnist.info.INFO["dermamnist"]["label"].values()


@register_dataset("oct", "multiclass")
class OCTDataset(NpzVisionDataset):
    classes = medmnist.info.INFO["octmnist"]["label"].values()


@register_dataset("retina", "multiclass")
class RetinaDataset(NpzVisionDataset):
    classes = medmnist.info.INFO["retinamnist"]["label"].values()


@register_dataset("breast", "binary")
class BreastDataset(NpzVisionDataset):
    classes = medmnist.info.INFO["breastmnist"]["label"].values()


@register_dataset("blood", "multiclass")
class BloodDataset(NpzVisionDataset):
    classes = medmnist.info.INFO["bloodmnist"]["label"].values()


def get_dataset_class(name: str) -> Type[NpzVisionDataset]:
    name = name.lower()
    try:
        cls = BUILTIN_DATASETS[name][0]
    except KeyError:
        raise ValueError(f"Unknown dataset {name}")
    return cls


def get_dataset(name: str, **config: Any) -> NpzVisionDataset:
    cls = get_dataset_class(name)
    return cls(**config)


def get_dataset_problem(name: str) -> str:
    name = name.lower()
    try:
        problem = BUILTIN_DATASETS[name][1]
    except KeyError:
        raise ValueError(f"Unknown dataset {name}")
    return problem
