import os
import cv2
import sys
import numpy as np


def checkFile(file_path):
    if not os.path.isfile(file_path):
        message = ' '.join(['File does not exist:', file_path])
        print(message)
        sys.exit(1)


def checkDevice(source, api=None):
    if not api:
        cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(source, api)
    if cap is None or not cap.isOpened():
        message = ' '.join(['File does not exist:', file_path])
        print(message)
        sys.exit(1)
    return cap


def loadData(file_path):
    checkFile(file_path)
    data = np.loadtxt(file_path, dtype=np.float32, delimiter=',')
    return data
