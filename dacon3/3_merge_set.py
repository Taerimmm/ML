import numpy as np
import cv2

# X_Train - numpy 

size = 5000
for j in range(10):
    a = []
    b = []      
    for i in range(j*5000, j*5000+size):
        img = cv2.imread('../Data/dirty_mnist_2nd/{0:05d}.png'.format(i), cv2.IMREAD_GRAYSCALE)
        denoised_img = cv2.fastNlMeansDenoising(img, None, 30, 15, 21)

        img = img / 255
        denoised_img = denoised_img / 255

        a.append(img)
        b.append(denoised_img)

    x_train = np.array(a)
    x_train_denoise = np.array(b)

    print(x_train)
    print(type(x_train))
    print(x_train.shape)

    print(x_train_denoise)
    print(type(x_train_denoise))
    print(x_train_denoise.shape)

    np.save('./dacon3/data/x_train_merge_{}.npy'.format(j+1), x_train)
    np.save('./dacon3/data/x_train_denoise_merge_{}.npy'.format(j+1), x_train_denoise)

    print(j+1, "End !!")

# X_Test - numpy

a = []
b = []
size = 5000
for i in range(size):
    img = cv2.imread('../Data/test_dirty_mnist_2nd/{0:05d}.png'.format(i+50000), cv2.IMREAD_GRAYSCALE)
    denoised_img = cv2.fastNlMeansDenoising(img, None, 30, 15, 21)

    img = img / 255
    denoised_img = denoised_img / 255

    a.append(img)
    b.append(denoised_img)

x_test = np.array(a)
x_test_denoise = np.array(b)

print(x_test)
print(type(x_test))
print(x_test.shape)

print(x_test_denoise)
print(type(x_test_denoise))
print(x_test_denoise.shape)


np.save('./dacon3/data/x_test_merge.npy', x_test)
np.save('./dacon3/data/x_test_denoise_merge.npy', x_test_denoise)
