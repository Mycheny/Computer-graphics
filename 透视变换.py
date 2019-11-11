import cv2
import numpy as np


def nothing(value):
    pass


if __name__ == '__main__':
    windowName = "remote control"
    cap = cv2.VideoCapture(0)
    cap.set(3, 2160 // 5)
    cap.set(4, 3840 // 5)
    cell_h = 64
    cell_w = 64
    cv2.namedWindow(windowName, 1)
    cv2.createTrackbar("center_x", windowName, 75, 100, nothing)
    cv2.createTrackbar("center_y", windowName, 75, 100, nothing)
    cv2.createTrackbar("angle", windowName, 50, 100, nothing)
    cv2.createTrackbar("scale_x", windowName, 25, 100, nothing)
    cv2.createTrackbar("scale_y", windowName, 25, 100, nothing)
    M_bak = np.zeros((3,3), np.float32)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for i in range(60):
            cv2.line(image, (0, i * cell_w), (image.shape[1], i * cell_w), (0, 255, 0))
            cv2.line(image, (i * cell_h, 0), (i * cell_h, image.shape[0]), (0, 255, 0))
        h, w = image.shape
        center_x = cv2.getTrackbarPos("center_x", windowName)
        center_y = cv2.getTrackbarPos("center_y", windowName)
        angle = cv2.getTrackbarPos("angle", windowName)
        scale_x = cv2.getTrackbarPos("scale_x", windowName)
        scale_y = cv2.getTrackbarPos("scale_y", windowName)
        center_x = int((center_x - 50) / 50 * w)
        center_y = int((center_y - 50) / 50 * h)
        angle = ((angle - 50) / 50 * 360 * 1) / 180 * np.pi
        scale_x = scale_x/25
        scale_y = scale_y/25
        # print(center_x, center_y, angle, scale_x, scale_y)
        M = np.array([[scale_x * np.cos(angle), scale_y * np.sin(angle),
                       (1 - scale_x * np.cos(angle)) * center_x - scale_x * center_y * np.sin(angle)],
                      [-scale_y * np.sin(angle), scale_y * np.cos(angle),
                       (1 - scale_y * np.cos(angle)) * center_y + scale_y * center_x * np.sin(angle)],
                      [0, 0, 1]]).astype(np.float32)
        if not (M_bak == M).all():
            print(M)
            print(M.min(), M.max(), "\n")
            M_bak = M
        dst1 = image
        dst = cv2.warpPerspective(image, M, (w, h), dst1)
        cv2.circle(dst, (center_x, center_y), 3, (255, 255, 255), 3)
        cv2.circle(dst, (center_x, center_y), 6, (0, 0, 0), 3)
        cv2.imshow("dst", dst)
        cv2.waitKey(1)
