import concurrent.futures

from PIL import Image
from pathlib import Path
from tqdm import tqdm

RESIZE_WIDTH = RESIZE_HEIGHT = 256
RESIZE_SIZE = (RESIZE_WIDTH, RESIZE_HEIGHT)

def resize_and_save(open_fp, save_fp):
    with Image.open(open_fp) as im:
        im_resized = im.resize(RESIZE_SIZE, Image.Resampling.BICUBIC)

        im_resized.save(save_fp)


fp_root = Path("/mnt/jbrockma/")
cxr8_root = fp_root / "CXR8"
cxr8_images_root = cxr8_root / "images" / "images"
chest_root = fp_root / "bachelor-thesis-images" / "chest"

cxr8_image_files = set(cxr8_images_root.iterdir())

tqdm_total = len(cxr8_image_files)

with tqdm(total=tqdm_total) as pbar:
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []

        def update_progress(_):
            pbar.update()

        for cxr8_image_file in cxr8_image_files:
            save_fp = chest_root / f"{cxr8_image_file.name}"
            future = executor.submit(resize_and_save, cxr8_image_file, save_fp)
            future.add_done_callback(update_progress)
            futures.append(future)

        concurrent.futures.wait(futures)
