import cv2
import numpy

# 暴力匹配
img1 = cv2.imread('imgs/1-1.jpg')
img2 = cv2.imread("imgs/1-2.jpg")


def show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


show("img1", img1)

