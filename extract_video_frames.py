import sys
import argparse

import cv2
print(cv2.__version__)


def extractImages(pathIn, pathOut, numIms):
    count = 0
    vidcap = cv2.VideoCapture(pathIn)
    success, image = vidcap.read()
    success = True
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (count*1000))    # added this line
        success, image = vidcap.read()
        save_path = pathOut + "\\frame%d.png" % count  # PNG FORMAT
        print('Read a new frame: ', success, f'Saving as:  {save_path}')
        cv2.imwrite(save_path, image)     # save frame
        count = count + 1
        if numIms and count > numIms:
            break


if __name__=="__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--pathIn", help="path to video", default=r"C:\Users\Nil\Downloads\Alternate Realities.mp4")
    a.add_argument("--pathOut", help="path to images", default=r"C:\Users\Nil\PycharmProjects\StyleGAN2\data\Alternate_realities_2400")
    a.add_argument("--numIms", help="number of images to extract", type=int, default=0)
    args = a.parse_args()
    print(args)
    extractImages(args.pathIn, args.pathOut, args.numIms)
