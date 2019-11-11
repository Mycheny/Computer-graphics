import numpy as np
import cv2

if __name__ == '__main__':
    a = np.array([[1, 0, 5],
                  [0, 1, 3],
                  [0, 0, 1]])
    b = np.array([[0.5, 0, 0],
                  [0, 0.5, 0],
                  [0, 0, 1]])
    c = np.array([[1, 0, -5],
                  [0, 1, -3],
                  [0, 0, 1]])
    point = np.array([[9],
                      [9],
                      [1]])
    d = a.dot(b).dot(c)
    res = d.dot(point)
    res1 = c.dot(point)
    res1 = b.dot(res1)
    res1 = a.dot(res1)
    print()
