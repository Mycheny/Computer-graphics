import numpy as np
import cv2


def nothing(value):
    print(value)


if __name__ == '__main__':
    # image = cv2.imread("a.png", 0)
    cap = cv2.VideoCapture(0)
    cap.set(3, 2160//3)
    cap.set(4, 3840//3)
    cell_h = 63
    cell_w = 63
    offset_h = 0
    offset_w = 0
    windowName = ""
    ok, frame = cap.read()
    if not ok:
        quit(0)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    r = 0
    max_h_r = image.shape[0]//4
    max_w_r = image.shape[1]//4
    cv2.namedWindow(windowName, 1)
    cv2.createTrackbar("h_r", windowName, r, max_h_r, nothing)
    cv2.createTrackbar("w_r", windowName, r, max_w_r, nothing)

    while 1:
        ok, frame = cap.read()
        if not ok:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for i in range(60):
            cv2.line(image, (0, i * cell_w), (image.shape[1], i * cell_w), (0, 255, 0))
            cv2.line(image, (i * cell_h, 0), (i * cell_h, image.shape[0]), (0, 255, 0))
        h, w = image.shape
        pts0 = np.float32([[0, 0], [0, h / 2], [w / 2, h / 2], [w / 2, 0]])
        pts00 = np.float32([[offset_w, offset_h], [0, h / 2], [w / 2, h / 2], [w / 2, 0]])
        M0 = cv2.getPerspectiveTransform(pts0, pts00)

        pts1 = np.float32([[0, 0], [0, h / 2], [w / 2, h / 2], [w / 2, 0]])
        pts11 = np.float32([[0, 0], [0, h / 2], [w / 2, h / 2], [w / 2 - offset_w, 0 + offset_h]])
        M1 = cv2.getPerspectiveTransform(pts1, pts11)

        pts2 = np.float32([[0, 0], [0, h / 2], [w / 2, h / 2], [w / 2, 0]])
        pts22 = np.float32([[0, 0], [0 + offset_w, h / 2 - offset_h], [w / 2, h / 2], [w / 2, 0]])
        M2 = cv2.getPerspectiveTransform(pts2, pts22)

        pts3 = np.float32([[0, 0], [0, h / 2], [w / 2, h / 2], [w / 2, 0]])
        pts33 = np.float32([[0, 0], [0, h / 2], [w / 2 - offset_w, h / 2 - offset_h], [w / 2, 0]])
        M3 = cv2.getPerspectiveTransform(pts3, pts33)

        dst0 = cv2.warpPerspective(image[:int(h / 2), :int(w / 2)], M0, (int(w / 2), int(h / 2)))
        dst1 = cv2.warpPerspective(image[:int(h / 2), int(w / 2):], M1, (int(w / 2), int(h / 2)))
        dst2 = cv2.warpPerspective(image[int(h / 2):, :int(w / 2)], M2, (int(w / 2), int(h / 2)))
        dst3 = cv2.warpPerspective(image[int(h / 2):, int(w / 2):], M3, (int(w / 2), int(h / 2)))
        offset_h = cv2.getTrackbarPos("h_r", windowName)
        offset_w = cv2.getTrackbarPos("w_r", windowName)
        cv2.imshow("", np.vstack((np.hstack((dst0, dst1)), np.hstack((dst2, dst3)))))
        cv2.waitKey(100)
