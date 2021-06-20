from pathlib import Path
import h5py
import numpy as np
import cv2
from tqdm import tqdm
import os

PREPROCESSED_PATH = Path("//ar-deeplearner/data_ro/nil/trained_models/LAX/data/preprocessed.h5")
DATA_SAVE_DIR = Path('C:/Users/Nil/PycharmProjects/StyleGAN2/data/LAX_3ch_256')


def prepr_to_single_im():
    preprocess = h5py.File(PREPROCESSED_PATH, mode='r')
    os.makedirs(str(DATA_SAVE_DIR), exist_ok=True)
    data = preprocess['images']
    for i, im in tqdm(enumerate(data)):
        name = DATA_SAVE_DIR / f'{str(i).zfill(6)}.png'
        # im = cv2.resize(im, None, fx=.5, fy=.5, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(name), (im*255))


def prepr_to_single_im_3ch():
    preprocess = h5py.File(PREPROCESSED_PATH, mode='r')
    os.makedirs(str(DATA_SAVE_DIR), exist_ok=True)
    data = preprocess['images']
    for i, im in tqdm(enumerate(data)):
        name = DATA_SAVE_DIR / f'{str(i).zfill(6)}.png'
        # im = cv2.resize(im, None, fx=.5, fy=.5, interpolation=cv2.INTER_NEAREST)
        im = np.stack((im, im, im), axis=2)
        cv2.imwrite(str(name), (im * 255))


if __name__ == '__main__':
    prepr_to_single_im_3ch()