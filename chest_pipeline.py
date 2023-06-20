import concurrent.futures
import numpy as np
from PIL.Image import Resampling

from constants import *
import pandas as pd
from tqdm import tqdm
from PIL import Image


def rename_columns(df, new_columns):
    return df.rename(columns=new_columns)


def keep_columns(df, columns):
    return df[columns]


def transform_to_dummy(df, column, sep):
    dummies = df[column].str.get_dummies(sep).rename(columns=str.lower)
    df_without_column = df.drop(columns=[column])
    return pd.concat([df_without_column, dummies], axis=1)


def drop_column(df, column):
    return df.drop(columns=[column])


def clean_image_data():
    image_data = pd.read_csv(cxr8_root / "Data_Entry_2017_v2020.csv")
    renaming_transformation = {
        "Image Index": "file_name",
        "Finding Labels": "findings",
    }
    columns_to_keep = [name for name in renaming_transformation.values()]
    cleaned_image_data = (
        image_data.pipe(rename_columns, new_columns=renaming_transformation)
        .pipe(keep_columns, columns=columns_to_keep)
        .pipe(transform_to_dummy, column="findings", sep="|")
        .pipe(drop_column, "no finding")
    )
    return cleaned_image_data


def resize_images():
    cxr8_images_root = cxr8_root / "images" / "images"

    cxr8_image_files = set(cxr8_images_root.iterdir())
    images_n = len(cxr8_image_files)

    def resize_image(image):
        with Image.open(image) as im:
            resized_im = im.resize(SIZE, Resampling.BICUBIC)
            # ensure grayscale
            im_to_save = resized_im.convert("L") if resized_im.mode != "L" else resized_im
            im_to_save.save(images_root / "chest" / image.name)

    with tqdm(desc="Resizing images", total=images_n) as pbar:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = set()

            def update_progress(_):
                pbar.update()

            for cxr8_image_file in cxr8_image_files:
                future = executor.submit(resize_image, cxr8_image_file)
                future.add_done_callback(update_progress)
                futures.add(future)

    exceptions = set()

    for future in futures:
        exception = future.exception()
        if exception:
            exceptions.add(exception)

    for exception in exceptions:
        print(exception)


def create_array(image_list_file, labels_file):
    with open(image_list_file) as f:
        image_file_names = [line.strip() for line in f.readlines()]

    n_images = len(image_file_names)

    labels = pd.read_csv(labels_file, index_col="file_name")
    n_labels = len(labels.columns)

    image_pixels = np.empty((n_images, HEIGHT, WIDTH), dtype=np.uint8)
    image_labels = np.empty((n_images, n_labels), dtype=np.uint8)

    chest_images_root = images_root / "chest"

    def write_to_array(image_file_name, i):
        with Image.open(chest_images_root / image_file_name) as im:
            arr = np.asarray(im)
            image_pixels[i] = arr
            image_labels[i] = labels.loc[image_file_name]

    with tqdm(desc="Writing to arrays", total=n_images) as pbar:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = set()

            def update_progress(_):
                pbar.update()

            for i, image_file_name in enumerate(image_file_names):
                future = executor.submit(write_to_array, image_file_name, i)
                future.add_done_callback(update_progress)
                futures.add(future)

            concurrent.futures.wait(futures)

            exceptions = set()

    for future in futures:
        exception = future.exception()
        if exception:
            exceptions.add(exception)

    for exception in exceptions:
        print(exception)

    return image_pixels, image_labels


def main():
    chest_image_labels= clean_image_data()

    chest_image_labels.to_csv(data_root / "chest" / "image-labels.csv", index=False)

    resize_images()

    test_images, test_labels = create_array(cxr8_root / "test_list.txt", data_root / "chest-image-data.csv")
    np.savez(root / "bachelor-thesis-npz" / "test-chest.npz", test_images=test_images, test_labels=test_labels)


if __name__ == '__main__':
    main()
