import numpy as np
import cv2

# X_Train - numpy 
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

np.save('./dacon3/data/x_train_merge.npy', x_train)

# X_Test - numpy
a = []
size = 50
for i in range(size):
    img = cv2.imread('../Data/test_dirty_mnist_2nd/{0:05d}.png'.format(i+50000))

    denoised_img = cv2.fastNlMeansDenoising(img, None, 30, 15, 21)

    a.append(denoised_img)

x_test = np.array(a)
x_test = x_test / 255
print(x_test)
print(type(x_test))
print(x_test.shape)

np.save('./dacon3/data/x_test_merge.npy', x_test)
