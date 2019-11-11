import cv2
import tensorflow as tf
import numpy as np


def nothing(*arg):
    pass


def rotate(image, angle):
    """
    旋转图片
    :param image:
    :param angle:
    :return:
    """
    PI = np.pi
    heightNew = int(
        image.shape[1] * np.abs(np.sin(angle * PI / 180)) + image.shape[0] * np.abs(np.cos(angle * PI / 180)))
    widthNew = int(
        image.shape[0] * np.abs(np.sin(angle * PI / 180)) + image.shape[1] * np.abs(np.cos(angle * PI / 180)))
    pt = (image.shape[1] / 2., image.shape[0] / 2.)
    # 旋转矩阵
    # --                      --
    # | ∂, β, (1 -∂) * pt.x - β * pt.y |
    # | -β, ∂, β * pt.x + (1 -∂) * pt.y |
    # --                      --
    # 其中 ∂=scale * cos(angle), β = scale * sin(angle)

    ### getRotationMatrix2D 的实现(多了平移) ###
    scale = 1
    a = scale * np.cos(angle * PI / 180)
    b = scale * np.sin(angle * PI / 180)
    r1 = np.array([[a, b, (1 - a) * pt[0] - b * pt[1] + (widthNew - image.shape[1]) / 2],
                   [-b, a, b * pt[0] + (1 - a) * pt[1] + (heightNew - image.shape[0]) / 2]])
    ### getRotationMatrix2D 的实现 ###

    r = cv2.getRotationMatrix2D(pt, angle, 1.0)  # 获取旋转矩阵(旋转中心(pt), 旋转角度(angle)， 缩放系数(scale)

    r[0, 2] += (widthNew - image.shape[1]) / 2
    r[1, 2] += (heightNew - image.shape[0]) / 2
    dst = cv2.warpAffine(image, r1, (widthNew, heightNew), cv2.INTER_LINEAR,
                         cv2.BORDER_CONSTANT)  # 进行仿射变换（输入图像, 2X3的变换矩阵, 指定图像输出尺寸, 插值算法标识符, 边界填充BORDER_REPLICATE)

    # pts = np.float32([[0, 0], [0, 1080], [1920, 1080], [1920, 0]])
    # pts1 = np.float32([[100, 0], [300, 1080], [1000, 1080], [1920, 0]])
    # M = cv2.getPerspectiveTransform(pts, pts1)
    # M1 = cv2.getAffineTransform(pts[:3,:], pts1[:3,:])
    # points = []
    # for i in range(image.shape[0]):
    #     for j in range(image.shape[1]):
    #         points.append([i, j])
    # points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    # points_new = cv2.perspectiveTransform(points, M)
    # dst = cv2.warpPerspective(image, M, (1920, 1080))
    return dst


if __name__ == '__main__':
    windowName = "erode"
    w, h = int(3840 / 4), int(2160 / 4)
    cap = cv2.VideoCapture("C:/Users/xiaoi/Pictures/Camera Roll/001.mp4")
    cap.set(3, 3840)
    cap.set(4, 2160)

    image_p = tf.placeholder(tf.uint8, (1, h, w, 1))
    kernel = tf.placeholder(tf.float32, (None, None, 1))
    input = tf.cast(image_p, tf.float32)
    erode = tf.nn.erosion2d(input, kernel, [1, 1, 1, 1], [1, 1, 1, 1], "SAME")
    dilate = tf.nn.dilation2d(input, kernel, [1, 1, 1, 1], [1, 1, 1, 1], "SAME")
    res = dilate - erode
    r = 1
    max_r = 100
    cv2.namedWindow(windowName, 1)
    cv2.createTrackbar("r", windowName, r, max_r, nothing)

    with tf.Session() as sess:
        angle = 30
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (w, h))
            image = np.resize(frame, (1, h, w, 1))

            r = cv2.getTrackbarPos("r", windowName)
            s = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1 * r + 1, 1 * r + 1))  # 乘数如果是单数画面会发生抖动
            k = np.expand_dims(s, -1).astype(np.float32)

            out = sess.run(res, feed_dict={image_p: image, kernel: k})
            show = out[0, ..., 0]
            show = (show - show.min()) / (show.max() - show.min())
            angle = angle % 360
            show = rotate(show, angle)
            cv2.imshow(windowName, show)
            # d = cv2.dilate(frame, s)
            # cv2.imshow(windowName, d)
            cv2.waitKey(0)
            angle += 1
