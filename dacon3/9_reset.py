import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

i = 92
image_path = '../data/dirty_mnist_2nd/{:05d}.png'.format(i)
img = cv2.imread(image_path)
img = np.where(img>=255, img, 0)
_img = cv2.dilate(img, kernel=np.ones((2,2), np.uint8), iterations=1)
_img = cv2.medianBlur(src=_img, ksize=5)
# _img = cv2.bilateralFilter(_img, 10, 50, 50)
_img = np.where(_img>=255, 255, 0)
_img = np.array(_img, dtype=np.uint8)
# denoised_img = cv2.fastNlMeansDenoising(img, None, 30, 7, 9)

cv2.imshow('before', img)
cv2.imshow("after", _img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# pd.read_csv('../data/mnist_data/train.csv')