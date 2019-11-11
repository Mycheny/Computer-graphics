import cv2
import numpy as np


def nothing(*arg):
    pass

if __name__ == '__main__':
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("C:/Users/xiaoi/Pictures/Camera Roll/001.mp4")
    cap.set(3, 3840)
    cap.set(4, 2160)
    image = cv2.imread("image/WIN_20191107_11_18_12_Pro.jpg", 0)
    image = cv2.resize(image, (960, 540))

    # 膨胀
    r = 1
    max_r = 100
    cv2.namedWindow("dilate", 1)
    cv2.createTrackbar("r", "dilate", r, max_r, nothing)
    while 1:
        ok, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (960, 540))
        d = image
        r = cv2.getTrackbarPos("r", "dilate")
        s = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))

        # d = cv2.erode(d, s)  # 腐蚀
        # d = cv2.dilate(d, s)  # 膨胀
        #
        # # 开运算
        # d = cv2.erode(d, s)
        # d = cv2.dilate(d, s)
        #
        # # # 闭运算
        # d = cv2.dilate(d, s)
        # d = cv2.erode(d, s)

        # # 顶帽运算
        # d = cv2.erode(d, s)
        # d = cv2.dilate(d, s)
        # d = image-d

        # 底帽运算
        # d = cv2.dilate(d, s)
        # d = cv2.erode(d, s)
        # d = image - d

        # 形态学边缘检测（形态学梯度）
        d = cv2.erode(d, s)
        d = cv2.dilate(image, s) - d

        cv2.imshow("dilate", d)
        if cv2.waitKey(5) == 27:
            break
    cv2.destroyAllWindows()
