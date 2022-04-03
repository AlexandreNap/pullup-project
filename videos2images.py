# -*- coding: utf-8 -*-

import cv2
import os
from os import listdir
from os.path import isfile, join


def main(i):
    path = os.getcwd()
    mypath = path + "\\data\\videos_buffer"
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    
    # os.makedirs(join(path, "data/images/images%d" % i))
    # count = 1
    for f in onlyfiles:
        os.makedirs(join(path, "data/images/images%d" % i))
        count = 1
        vidcap = cv2.VideoCapture(join(mypath, f))
        success, image = vidcap.read()
        while success:
            cv2.imwrite("data/images/images%d/im_%d.jpg" % (i, count), image)
            success, image = vidcap.read()
            print('Saved image ', count)
            count += 1
        i += 1


if __name__ == "__main__":
    main(15)
