import os
import sys

from pathlib import Path

import argparse

import concurrent.futures
import random

import numpy as np

import pandas as pd
from tqdm import tqdm
from PIL import Image

SPLITS = ["train", "val", "test"]


def resolve_paths(args):
    for name in ["out_dest", "source", "split_info", "oct_mnist"]:
        path_str = getattr(args, name)
        path_str = os.path.expandvars(path_str)
        path = Path(path_str).expanduser().resolve()
        setattr(args, name, path)


def main(args):
    out_is_dir = args.out_dest.endswith("/")

    resolve_paths(args)

    if args.split_info.is_dir():
        args.split_info /= "octmnist_split_info.csv"

    if out_is_dir:
        args.out_dest.mkdir(parents=True, exist_ok=True)
    else:
        args.out_dest.parent.mkdir(parents=True, exist_ok=True)

    split_info = pd.read_csv(args.split_info, index_col="image_id")
    split_value_counts = split_info["split"].value_counts()

    n_samples_of_split = {SPLIT: split_value_counts[SPLIT] for SPLIT in SPLITS}

    images_of_split = {}
    for SPLIT in SPLITS:
        n_samples = n_samples_of_split[SPLIT]
        images_of_split[SPLIT] = np.empty((n_samples, args.size, args.size), dtype=np.uint8)

    info_of_image = {im_id: (split, idx) for im_id, split, idx in split_info.itertuples()}

    with tqdm(desc="Processing", total=len(split_info), unit="pic") as pbar:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = set()

            def process(_fp, _info):
                _split, _index = _info

                with Image.open(_fp) as im:
                    if im.mode != "L":
                        im = im.convert("L")
                    im = im.resize((args.size, args.size), Image.BICUBIC)

                    images_of_split[_split][_index] = np.asarray(im)

            for child in args.source.rglob("*"):
                if child.is_file():
                    try:
                        info = info_of_image.pop(child.stem)
                    except KeyError:
                        pbar.write(f"Skipping {child.name}")
                        continue

                    future = executor.submit(process, child, info)
                    future.add_done_callback(lambda _: pbar.update())
                    futures.add(future)

            missing_ims = list(info_of_image.keys())

            if not len(missing_ims) == 0:
                pbar.write(f"The dataset at {args.source} is incomplete.")
                if len(missing_ims) == 1:
                    pbar.write(f"Missing image with id {missing_ims[0]}")
                elif 1 < len(info_of_image) < 6:
                    im_ids_str = ", ".join(missing_ims[:-1]) + f", and {missing_ims[-1]}"
                    pbar.write(f"Missing files with ids {im_ids_str}")
                else:
                    random_im_ids = random.sample(missing_ims, 5)
                    random_im_ids_str = ", ".join(random_im_ids[:-1]) + f", and {random_im_ids[-1]}"
                    pbar.write(f"Missing {len(missing_ims)} images such as {random_im_ids_str}")
                sys.exit(1)

            concurrent.futures.wait(futures)

    oct_mnist = np.load(args.oct_mnist)
    labels_of_split = {SPLIT: oct_mnist[f"{SPLIT}_labels"] for SPLIT in SPLITS}

    name_to_array = {}
    for data_name, data_of_split in [("images", images_of_split), ("labels", labels_of_split)]:
        for SPLIT in SPLITS:
            name_to_array[f"{SPLIT}_{data_name}"] = data_of_split[SPLIT]

    if args.out_dest.is_dir():
        args.out_dest /= f"oct-{args.size}.npz"

    print(f"Saving {args.out_dest}. This might take a while ...")

    np.savez_compressed(args.out_dest, **name_to_array)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="OCTMnistAdjuster",
                                     description="Resize the Optical Coherence Tomography (OCT) dataset to any size and save them to an npz file the same way as with OCTMnist")

    parser.add_argument("source", type=str, help="directory holding the (384-1536) x (277-512) pixel images from the OCT dataset")
    parser.add_argument("split_info", type=str,
                        help="directory or the the csv file directly containing columns split, column and image_id information about the OCTMnist "
                             "dataset")
    parser.add_argument("--octmnist", "--oct-mnist", default=os.path.join("~", ".medmnist", "octmnist.npz"), type=str,
                        dest="oct_mnist", help="octmnist.npz file from which the labels are copied")
    parser.add_argument("-s", "--size", default=224, type=int,
                        help="the size to which the source images are resized to. 224 by default")
    parser.add_argument("-o", "--out", "--out-dest", default=".", type=str, dest="out_dest",
                        help="destination of the output npz file")

    main(parser.parse_args())
