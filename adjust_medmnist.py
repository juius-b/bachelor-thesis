import os
import sys
from dataclasses import dataclass

from pathlib import Path

import argparse

import concurrent.futures
import random
from typing import Dict

import numpy as np

import pandas as pd
from tqdm import tqdm
from PIL import Image


SPLITS = ['train', 'val', 'test']


@dataclass
class Config:
    split_info: Path
    size: int
    mode: str
    source: Path
    med_mnist: str
    out_dest: Path
    flag: str


@dataclass
class Parameters:
    mode: str
    size: int
    images_of_split: Dict[str, np.ndarray]


def process(fp, info, params: Parameters):
    split, index = info

    with Image.open(fp) as im:
        if im.mode != params.mode:
            im = im.convert(params.mode)
        im = im.resize((params.size, params.size), Image.BICUBIC)

        params.images_of_split[split][index] = np.asarray(im)


def main(cfg: Config):
    split_info = pd.read_csv(cfg.split_info, index_col='image_id')
    split_value_counts = split_info['split'].value_counts()

    n_samples_of_split = {SPLIT: split_value_counts[SPLIT] for SPLIT in SPLITS}

    med_mnist = np.load(cfg.med_mnist)

    if cfg.mode == 'auto':
        # RGB datasets require an extra dimension to carry the information of three different color channels
        cfg.mode = 'L' if len(med_mnist['val_images'].shape) == 3 else 'RGB'
        print(f'Chose mode {cfg.mode} based on flag {cfg.flag}')

    images_of_split = {}
    for SPLIT in SPLITS:
        n_samples = n_samples_of_split[SPLIT]
        images_of_split[SPLIT] = np.empty((n_samples, cfg.size, cfg.size, (3 if cfg.mode == 'RGB' else 1)),
                                          dtype=np.uint8).squeeze()

    info_of_image = {im_id: (split, idx) for im_id, split, idx in split_info.itertuples()}

    futures = set()
    params = Parameters(cfg.mode, cfg.size, images_of_split)
    with tqdm(desc='Processing', total=len(split_info), unit='pic') as pbar:
        with concurrent.futures.ProcessPoolExecutor() as executor:  # to avoid Python's GIL
            for child in cfg.source.rglob('*'):
                if child.is_file():
                    try:
                        info = info_of_image.pop(child.stem)
                    except KeyError:
                        pbar.write(f'Skipping {child.name}')
                        continue

                    future = executor.submit(process, child, info, params)
                    future.add_done_callback(lambda _: pbar.update())
                    futures.add(future)

            missing_ims = list(info_of_image.keys())

            if not len(missing_ims) == 0:
                pbar.write(f'The dataset at {cfg.source} is incomplete.')
                if len(missing_ims) == 1:
                    pbar.write(f'Missing image with id {missing_ims[0]}')
                elif 1 < len(info_of_image) < 6:
                    im_ids_str = ', '.join(missing_ims[:-1]) + f', and {missing_ims[-1]}'
                    pbar.write(f'Missing files with ids {im_ids_str}')
                else:
                    random_im_ids = random.sample(missing_ims, 5)
                    random_im_ids_str = ', '.join(random_im_ids[:-1]) + f', and {random_im_ids[-1]}'
                    pbar.write(f'Missing {len(missing_ims)} images such as {random_im_ids_str}')
                sys.exit(5)

            concurrent.futures.wait(futures)

    print("Checking for exception that might have occurred during processing ...")
    exceptions = [future.exception() for future in futures if future.exception()]

    if not len(exceptions) == 0:
        print(f'Processing the source dataset was not without exceptions.')
        if len(exceptions) > 5:
            print('Multiple exceptions occurred such as the following.')
            for exception in random.sample(exceptions, 5):
                print(exception)
        else:
            for exception in exceptions:
                print(exception)
        sys.exit(6)
    else:
        print("No exceptions occurred during processing")

    labels_of_split = {SPLIT: med_mnist[f'{SPLIT}_labels'] for SPLIT in SPLITS}

    name_to_array = {}
    for data_name, data_of_split in [('images', images_of_split), ('labels', labels_of_split)]:
        for SPLIT in SPLITS:
            name_to_array[f'{SPLIT}_{data_name}'] = data_of_split[SPLIT]

    if cfg.out_dest.is_dir():
        cfg.out_dest /= f'{cfg.flag}-{cfg.size}.npz'

    print(f'Saving to {cfg.out_dest}. This might take a while ...')

    np.savez_compressed(cfg.out_dest, **name_to_array)


def resolve_paths(args):
    for name in {'med_mnist', 'out_dest', 'source', 'split_info'}:
        path_str = getattr(args, name)
        path_str = os.path.expandvars(path_str)
        path = Path(path_str).expanduser().resolve()
        setattr(args, name, path)


def validate_args(args):
    out_is_dir = args.out_dest.endswith('/')

    resolve_paths(args)

    if not args.source.exists():
        print(f'The directory of the source dataset {args.source} does not exist')
        sys.exit(1)

    if not args.flag:
        if args.split_info.is_file():
            args.flag = args.split_info.name.split('mnist')[0]
        elif args.med_mnist.is_file():
            args.flag = args.med_mnist.name.split('mnist')[0]
        else:
            print('Cannot specify the locations of the split info and the npz file of the MedMNIST npz file as '
                  'directories without specifying the flag')
            sys.exit(2)
    else:
        args.flag = args.flag.lower()

    if args.split_info.is_dir():
        args.split_info /= f'{args.flag}mnist_split_info.csv'
    if not args.split_info.exists():
        print(f'{args.split_info} does not exist')
        sys.exit(3)

    if args.med_mnist.is_dir():
        args.med_mnist /= f'{args.flag}mnist.npz'
    if not args.med_mnist.exists():
        print(f'{args.med_mnist} does not exist')
        sys.exit(4)

    if out_is_dir:
        args.out_dest.mkdir(parents=True, exist_ok=True)
    else:
        args.out_dest.parent.mkdir(parents=True, exist_ok=True)

    return Config(
        args.split_info,
        args.size,
        args.mode,
        args.source,
        args.med_mnist,
        args.out_dest,
        args.flag
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='MedMNISTAdjuster',
                                     description='Resize any source dataset of MedMNIST datasets to any size and save '
                                                 'them to an npz file with the labels in the same order as the '
                                                 'original MedMNIST dataset.')

    parser.add_argument('source', type=str,
                        help='directory holding the source images from the MedMNIST dataset at any depth')
    parser.add_argument('split_info', type=str,
                        help='the csv file containing columns split, column and image_id information about the '
                             'MedMNIST dataset or the parent directory')
    parser.add_argument('--medmnist', '--med-mnist', default='~/.medmnist', type=str, dest='med_mnist',
                        help='MedMNIST npz file from which the labels are copied or the parent directory of the file. '
                             'Defaults to ~/.medmnist')
    parser.add_argument('--flag', default=None, type=str,
                        help='flag of the dataset. E.g., "path" for PathMNIST. Can be deduced from split_info and '
                             'the original MedMNIST npz file if at least one is referencing a file and following the '
                             'naming convention')
    parser.add_argument('-s', '--size', default=224, type=int,
                        help='the size to which the source images are resized to. 224 by default')
    parser.add_argument('-m', '--mode', type=str, default='auto',
                        help='mode of the image. "L" for grayscale, "RGB" for rgb and "auto" to choose based on '
                             'the flag. "auto" by default')
    parser.add_argument('-o', '--out', '--out-dest', default='.', type=str, dest='out_dest',
                        help='destination of the output npz file')

    parsed_args = parser.parse_args()
    validated_args = validate_args(parsed_args)
    main(validated_args)
