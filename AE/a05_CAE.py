# 4번 copy 해서 붙혀넣기
# CNN 구성

import numpy as np
from tensorflow.keras.datasets import mnist

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255
x_test = x_test.reshape(10000, 28, 28, 1)/255

# print(x_train[0])
# print(x_test[0])  

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Conv2DTranspose, Input

def autoencoder_2():
    inputs = Input(shape=(28,28,1))
    layer1 = Conv2D(64, (2,2), strides=2, padding='same')(inputs)
    layer1_1 = layer1

    layer2 = Conv2D(128, (2,2), strides=2, padding='same')(layer1)
    layer2_2 = layer2

    layer3 = Conv2DTranspose(64, (2,2), strides=2, padding='same')(layer2+layer2_2)
    layer4 = Conv2DTranspose(1, (2,2), strides=2, padding='same')(layer3+layer1_1)

    outputs = layer4

    model = Model(inputs=inputs, outputs=outputs)

    return model

model = autoencoder_2()

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x_train, x_train, epochs=10)

output = model.predict(x_test)

import matplotlib.pyplot as plt
import random
fig, (row1, row2) = plt.subplots(2, 5, figsize=(20,7))

# 이미지 다섯 개를 무작위로 고른다.
random_images = random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다.

for i, ax in enumerate(row1):
    ax.imshow(x_test[random_images[i]].reshape(28,28), cmap='gray')
    
    if i == 0:
        ax.set_ylabel('INPUT', size=20)
    
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

for i, ax in enumerate(row2):
    ax.imshow(output[random_images[i]].reshape(28,28), cmap='gray')
    
    if i == 0:
        ax.set_ylabel('OUTPUT', size=20)
    
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()