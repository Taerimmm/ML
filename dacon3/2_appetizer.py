import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import cv2

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# png 파일 -> np.array() 로 변경 ?

# 노이즈 제거 fastNlMeansDenoising 사용 ?

img = cv2.imread('../Data/dirty_mnist_2nd/42672.png')

denoised_img = cv2.fastNlMeansDenoising(img, None, 30, 15, 21)

print(type(denoised_img))
print(img.shape)
print(denoised_img.shape)
# cv2.imshow('before', img)
# cv2.imshow("after", denoised_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

print(np.max(denoised_img), np.min(denoised_img))

a = []
size = 50
for i in range(size):
    img = cv2.imread('../Data/dirty_mnist_2nd/{0:05d}.png'.format(i))

    denoised_img = cv2.fastNlMeansDenoising(img, None, 30, 15, 21)

    a.append(denoised_img)

x_train = np.array(a)
x_train = x_train / 255
print(x_train)
print(type(x_train))
print(x_train.shape)

y_train = pd.read_csv('./dacon3/data/dirty_mnist_2nd_answer.csv', index_col=0, header=0)

y_train = y_train[:size]

print(y_train)
print(y_train.shape)

steps = 20
kfold = KFold(n_splits=steps, random_state=45, shuffle=True)

# # 2. 모델

# inputs = Input(shape=x_train.shape[1:])
# layer = 




# outputs = Dense(26, activation='sigmoid')(layer)

# model = Model(inputs=inputs, outputs=outputs)


# model.compile(loss='')