from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import os
import argparse

DATASET_DIR = r"C:\Users\Nil\PycharmProjects\StyleGAN2\data\ImaginaryLandscapes_top"
DATA_SAVE_DIR = r"C:\Users\Nil\PycharmProjects\StyleGAN2\data\ImaginaryLandscapes_top\256"


def main(dataset_dir, data_save_dir, resolution, inter=cv2.INTER_AREA):
    os.makedirs(str(data_save_dir), exist_ok=True)
    for filename in tqdm(os.listdir(dataset_dir)):
        img = cv2.imread(str(dataset_dir / filename))
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
            cv2.imwrite(str(data_save_dir / filename), cropped_im)


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument("--pathIn", help="path to video", type=str, default=DATASET_DIR)
    a.add_argument("--pathOut", help="path to images", type=str, default=DATA_SAVE_DIR)
    a.add_argument("--resolution", help="number of images to extract", type=int, default=256)
    args = a.parse_args()
    main(Path(args.pathIn), Path(args.pathOut), args.resolution)
