import numpy as np
from torch.utils.data import Dataset
from pathlib import Path


class ChestDataset(Dataset):
    def __init__(self, split):
        npz_file = Path("/mnt/jbrockma/bachelor-thesis-npz")
        arrays = np.load(npz_file)
        self.images = arrays[f"{split}_images"]
        self.labels = arrays[f"{split}_labels"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return self.images[item], self.labels[item]