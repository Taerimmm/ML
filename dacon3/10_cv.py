import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2


i = 92
image_path = '../data/dirty_mnist_2nd/{:05d}.png'.format(i)
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = np.where(img>=255, img, 0)
img = cv2.dilate(img, kernel=np.ones((2,2), np.uint8), iterations=1)
img = cv2.medianBlur(src=img, ksize=5)


cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# img, contours , hierachy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

#threshold를 이용하여 binary image로 변환
ret, thresh = cv2.threshold(img,127,255,0)

#contours는 point의 list형태. 예제에서는 사각형이 하나의 contours line을 구성하기 때문에 len(contours) = 1. 값은 사각형의 꼭지점 좌표.
#hierachy는 contours line의 계층 구조
contours, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
image = cv2.drawContours(img, contours, -1, (0,255,0), 3)

cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()