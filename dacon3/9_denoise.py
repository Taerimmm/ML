import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    '../data', 
    classes=['dirty_mnist_2nd'],
    batch_size=50000, 
    target_size=(64, 64), 
    color_mode='grayscale',
    class_mode=None,
    shuffle=False)

for i in train_generator:
    x_train = i
    break
print(x_train.shape)

img = x_train[0]
img = np.where(img>=255, img, 0)
img = cv2.dilate(img, kernel=np.ones((2,2), np.uint8), iterations=1)
img = cv2.medianBlur(scr=img, ksize=5)

'''
i = 92
image_path = '../data/dirty_mnist_2nd/{:05d}.png'.format(i)
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = np.where(img>=255, img, 0)
_img = cv2.dilate(img, kernel=np.ones((2,2), np.uint8), iterations=1)
_img = cv2.medianBlur(src=_img, ksize=5)
# _img = cv2.bilateralFilter(_img, 10, 50, 50)
# _img = np.where(_img>=255, img, 0)
# _img = np.array(_img, dtype=np.uint8)
# denoised_img = cv2.fastNlMeansDenoising(img, None, 30, 7, 9)
print(_img.shape)

cv2.imshow('before', img)
cv2.imshow("after", _img)
cv2.waitKey(0)
cv2.destroyAllWindows()

contour, hierachy = cv2.findContours(_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

cv2.drawContours(_img, [contour[0]], 0, (0, 0, 255), 2)
cv2.imshow(_img)
cv2.destroyAllWindows()
'''