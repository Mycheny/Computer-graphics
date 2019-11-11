import numpy as np
import cv2

if __name__ == '__main__':
    w, h = 640, 480
    angle = np.arange(0, 360, 1)
    scale = np.arange(0.1, 2, 0.1)
    center_x = np.arange(0, w, 10)
    center_y = np.arange(0, h, 1)

