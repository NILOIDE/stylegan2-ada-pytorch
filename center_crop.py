from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import os

DATASET_DIR = Path(r"C:\Users\Nil\PycharmProjects\StyleGAN2\data\ImaginaryLandscapes_top")
DATA_SAVE_DIR = Path(r"C:\Users\Nil\PycharmProjects\StyleGAN2\data\ImaginaryLandscapes_top\256")


def main(resolution=256, inter=cv2.INTER_AREA):
    os.makedirs(str(DATA_SAVE_DIR), exist_ok=True)
    for filename in tqdm(os.listdir(DATASET_DIR)):
        img = cv2.imread(str(DATASET_DIR / filename))
        if img is not None:
            assert img.shape[0] > 3
            (h, w) = img.shape[:2]
            if h < w:
                scale_factor = resolution / h
                dim = (int(np.ceil(w * scale_factor)), resolution)
            else:
                scale_factor = resolution / w
                dim = (resolution, int(np.ceil(h * scale_factor)))
            resized_img = cv2.resize(img, dim, interpolation=inter)
            crop_origin = ((resized_img.shape[0] - resolution) // 2, (resized_img.shape[1] - resolution) // 2)
            cropped_im = resized_img[crop_origin[0]:crop_origin[0]+resolution,
                                     crop_origin[1]:crop_origin[1]+resolution]
            assert cropped_im.shape == (resolution, resolution, 3),\
                f"Expected {(resolution, resolution, 3)}, received {cropped_im.shape}"
            cv2.imwrite(str(DATA_SAVE_DIR / filename), cropped_im)


if __name__ == '__main__':
    main()
