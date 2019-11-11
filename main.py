import math

import cv2
import numpy as np

point1 = np.expand_dims(np.array([[-25],[-25],[1]]), 0)
point2 = np.expand_dims(np.array([[-25],[25],[1]]), 0)
point3 = np.expand_dims(np.array([[25],[-25],[1]]), 0)
point4 = np.expand_dims(np.array([[25],[25],[1]]), 0)
points = np.concatenate((point1, point2, point3, point4))

scale_x = 1
scale_y = 1.5
angle = 0
translationx_x = 0
translationx_y = 0
shear_x = 2
shear_y = 0
matrix = np.array([[scale_x * math.cos(angle*math.pi/180), -math.sin(angle*math.pi/180)*shear_x, translationx_x],
                   [math.sin(angle*math.pi/180)*shear_y, scale_y * math.cos(angle*math.pi/180), translationx_y],
                   [0, 0, 1]])

canvas = np.zeros((200, 200, 3))
for point in points:
    p = np.asarray(np.dot(matrix, point), np.uint8)
    point += 100
    p += 100
    cv2.line(canvas, (0, 100), (200, 100), (125, 125, 125))
    cv2.line(canvas, (100, 0), (100, 200), (125, 125, 125))
    cv2.circle(canvas, tuple(point[:2, 0].tolist()), 2, (255, 255, 255))
    cv2.circle(canvas, tuple(p[:2, 0].tolist()), 3, (0, 0, 255))
cv2.imshow("canvas", canvas)
cv2.waitKey(0)
