# 2번 copy 해서 붙혀넣기
# 딥하게 구성
# 2개의 모델을 만드는데 
# 하나는 원칙적 오토인코더
# 다른 하나는 랜덤하게 만들고 싶은데로 히든을 구성
# 2개의 성능 비교       

import numpy as np
from tensorflow.keras.datasets import mnist

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 784).astype('float32')/255
x_test = x_test.reshape(10000, 784)/255

# print(x_train[0])
# print(x_test[0])  

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

# 대칭 O
def autoencoder_1():
    model = Sequential()
    model.add(Dense(392, input_shape=(784,), activation='relu'))
    model.add(Dense(196, activation='relu'))
    model.add(Dense(98, activation='relu'))
    model.add(Dense(49, activation='relu'))
    model.add(Dense(98, activation='relu'))
    model.add(Dense(196, activation='relu'))
    model.add(Dense(392, activation='relu'))
    model.add(Dense(units=784, activation='sigmoid'))

    return model

# 대칭 X
def autoencoder_2():
    model = Sequential()
    model.add(Dense(49, input_shape=(784,), activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(98, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(196, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(392, activation='relu'))
    model.add(Dense(units=784, activation='sigmoid'))

    return model

for i in range(2):
    model = globals()['autoencoder_{}'.format(i+1)]()

    model.summary()

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(x_train, x_train, epochs=10)

    globals()['output_{}'.format(i+1)] = model.predict(x_test)


import matplotlib.pyplot as plt
import random
fig, (row1, row2, row3) = plt.subplots(3, 5, figsize=(20,7))

# 이미지 다섯 개를 무작위로 고른다.
random_images = random.sample(range(output_1.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다.

for i, ax in enumerate(row1):
    ax.imshow(x_test[random_images[i]].reshape(28,28), cmap='gray')
    
    if i == 0:
        ax.set_ylabel('INPUT', size=20)
    
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

for i, ax in enumerate(row2):
    ax.imshow(output_1[random_images[i]].reshape(28,28), cmap='gray')
    
    if i == 0:
        ax.set_ylabel('OUTPUT_1', size=20)
    
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

for i, ax in enumerate(row3):
    ax.imshow(output_2[random_images[i]].reshape(28,28), cmap='gray')
    
    if i == 0:
        ax.set_ylabel('OUTPUT_2', size=20)
    
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()
