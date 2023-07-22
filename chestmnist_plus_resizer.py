import os

from pathlib import Path

import argparse

import concurrent.futures
import numpy as np

import pandas as pd
from tqdm import tqdm
from PIL import Image

SPLITS = ["train", "val", "test"]


def main(args):
    for name in ["source", "split_info", "chest_mnist", "out_dest"]:
        path_str = getattr(args, name)
        path_str = os.path.expandvars(path_str)
        path = Path(path_str).expanduser().resolve()
        is_dir = path.is_dir()
        exists = path.exists()
        is_file = path.is_file()
        if path.is_dir() and not path.exists():
            if name == "source":
                raise FileNotFoundError("Directory for the source images of the CXR8 dataset not found")
            else:  # out_dest
                path.mkdir()
        elif path.is_file():
            with open(path):
                pass
        setattr(args, name, path)
    split_info = pd.read_csv(args.split_info)
    split_value_counts = split_info["split"].value_counts()

    n_samples_of_split = {SPLIT: split_value_counts[SPLIT] for SPLIT in SPLITS}

    images_of_split = {}
    for SPLIT in SPLITS:
        n_samples = n_samples_of_split[SPLIT]
        images_of_split[SPLIT] = np.empty((n_samples, args.size, args.size), dtype=np.uint8)

    with tqdm(desc="Preprocessing", total=len(split_info)) as pbar:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = set()

            def preprocess(_info):
                _, _split, _index, _image_id = _info

                fp = args.source / f"{_image_id}.png"
                with Image.open(fp.expanduser().resolve()) as im:
                    if im.mode != "L":
                        im = im.convert("L")
                    im = im.resize((args.size, args.size), Image.BICUBIC)

                    images_of_split[_split][_index] = np.asarray(im)

            for info in split_info.itertuples():
                future = executor.submit(preprocess, info)
                future.add_done_callback(lambda _: pbar.update())
                futures.add(future)

            concurrent.futures.wait(futures)

    chest_mnist = np.load(args.chest_mnist)
    labels_of_split = {SPLIT: chest_mnist[f"{SPLIT}_labels"] for SPLIT in SPLITS}

    name_to_array = {}
    for data_name, data_of_split in [("images", images_of_split), ("labels", labels_of_split)]:
        for SPLIT in SPLITS:
            name_to_array[f"{SPLIT}_{data_name}"] = data_of_split[SPLIT]

    save_file = args.out_dest if args.out_dest.is_file() else args.out_dest / f"chest_{args.size}.npz"

    print(f"Saving {save_file}. This might take a while ...")

    np.savez_compressed(save_file, kwds=name_to_array)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="ChestMNIST+Resizer", description="Resize the ChestX-ray8 (CXR8) dataset to "
                                                                            "any size and save them to an npz file")

    parser.add_argument("source", type=str, help="directory holding the 1024 x 1024 pixel images "
                                                                    "from CXR8 dataset")
    parser.add_argument("split_info", type=str, help="csv file containing columns split, column and image_id information about the ChestMNIST dataset")
    parser.add_argument("--chestmnist", default=os.path.join("~", ".medmnist", "chestmnist.npz"), type=str,
                        dest="chest_mnist", help="chestmnist.npz file from which the labels are copied")
    parser.add_argument("-s", "--size", default=224, type=int,
                        help="the size to which the source images are resized to")
    parser.add_argument("-o", "--out", "--out-dest", default=".", type=str, dest="out_dest",
                        help="destination of the output npz file")

    main(parser.parse_args())
