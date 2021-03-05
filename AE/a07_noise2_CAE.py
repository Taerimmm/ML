import numpy as np
from tensorflow.keras.datasets import mnist

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255
x_test = x_test.reshape(10000, 28, 28, 1)/255

x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)
x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Conv2DTranspose, Input

def autoencoder():
    inputs = Input(shape=(28,28,1))
    layer1 = Conv2D(64, (2,2), strides=2, padding='same')(inputs)
    layer1_ = layer1

    layer2 = Conv2D(128, (2,2), strides=2, padding='same')(layer1)
    layer2_ = layer2

    layer3 = Conv2DTranspose(64, (2,2), strides=2, padding='same')(layer2 + layer2_)

    layer4 = Conv2DTranspose(1, (2,2), strides=2, padding='same')(layer3 + layer1_)
    
    outputs = layer4

    model = Model(inputs=inputs, outputs=outputs)

    return model

model = autoencoder()

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x_train_noised, x_train, epochs=10)

output = model.predict(x_test_noised)


from matplotlib import pyplot as plt
import random
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10), (ax11, ax12, ax13, ax14, ax15)) = plt.subplots(3, 5, figsize=(20,7))

# 이미지 5개를 무작위로 고른다.
random_imgs = random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다!!
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_imgs[i]].reshape(28,28), cmap='gray')

    if i == 0:
        ax.set_ylabel('INPUT', size=20)

    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 잡음을 넣은 이미지
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[random_imgs[i]].reshape(28,28), cmap='gray')

    if i == 0:
        ax.set_ylabel('Noise', size=20)

    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더가 출력한 이미지
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(output[random_imgs[i]].reshape(28,28), cmap='gray')

    if i == 0:
        ax.set_ylabel('OUTPUT', size=20)

    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()