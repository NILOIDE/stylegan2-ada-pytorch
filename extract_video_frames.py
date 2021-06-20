import sys
import argparse
from pathlib import Path
import tqdm
import math

import cv2
print('cv2 version:', cv2.__version__)


def extractImages(pathIn, pathOut, maxIms):
    vidcap = cv2.VideoCapture(pathIn)
    fps = vidcap.get(cv2.CAP_PROP_FPS)  # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = int(math.ceil(frame_count / fps))
    delta_s = 1

    success, image = vidcap.read()
    print('Saving images as: ', str(Path(pathOut) / "frame{s:07}.png"))
    s = 0
    for s in tqdm.tqdm(range(0, duration_s if maxIms <= 0 else maxIms, delta_s)):  # Iterate over seconds
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (s*1000))    # added this line
        success, image = vidcap.read()
        save_path = str(Path(pathOut) / f"frame{s:07}.png")  # PNG FORMAT
        if not success:
            print('Something went wrong. Frame reading was not a success.')
            quit()
        cv2.imwrite(save_path, image)     # save frame
    print(f'Saved {s+1} images')


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--pathIn", help="path to video", default=r"C:\Users\Nil\Downloads\Alternate Realities.mp4")
    a.add_argument("--pathOut", help="path to images", default=r"C:\Users\Nil\PycharmProjects\StyleGAN2\data\Alternate_realities_2400")
    a.add_argument("--maxIms", help="number of images to extract", type=int, default=20)
    args = a.parse_args()
    print(args)
    extractImages(args.pathIn, args.pathOut, args.maxIms)
