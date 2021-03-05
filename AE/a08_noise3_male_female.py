# [실습]
# keras67_1 남자 여자에 noise를 넣어서 제거하시오.

# 실습
# 남자 여자 구별
# ImageDataGenerator / fit_generator 사용해서 완성

import numpy as np
from matplotlib import pyplot as plt
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose, MaxPooling2D, Flatten, Dropout, BatchNormalization, Input
from tensorflow.keras.layers import LeakyReLU, ReLU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

x_train = np.load('../data/image/brain/npy/k67_train_x.npy')
x_test = np.load('../data/image/brain/npy/k67_val_x.npy')

print(x_train.shape) # (1389, 150, 150, 3)
print(x_test.shape)   # (347, 150, 150, 3)

# plt.imshow(x_test[0])
# plt.show()

x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)

x_train_noised = np.clip(x_train_noised, a_min = 0, a_max = 1)
x_test_noised = np.clip(x_test_noised, a_min = 0, a_max = 1)

def autoencoder():
    inputs = Input(shape=(150,150,3))
    layer1 = Conv2D(64, (3,3), strides=2, padding='same')(inputs)
    layer1_ = BatchNormalization()(layer1)
    layer1 = LeakyReLU()(layer1_)

    layer2 = Conv2D(128, (3,3), strides=2, padding='valid')(layer1)
    layer2_ = BatchNormalization()(layer2)
    layer2 = LeakyReLU()(layer2_)

    layer3 = Conv2D(256, (3,3), strides=2, padding='valid')(layer2)
    layer3_ = BatchNormalization()(layer3)
    layer3 = LeakyReLU()(layer3_)

    layer4 = Conv2D(512, (3,3), strides=2, padding='same')(layer3)
    layer4_ = BatchNormalization()(layer4)
    layer4 = LeakyReLU()(layer4_)

    layer5 = Conv2DTranspose(256, (3,3), strides=2, padding='same')(layer4)
    layer5 = BatchNormalization()(layer5)
    layer5 = layer5 + layer3_
    layer5 = Dropout(0.5)(layer5)
    layer5 = ReLU()(layer5)

    layer6 = Conv2DTranspose(128, (3,3), strides=2, padding='valid')(layer5)
    layer6 = BatchNormalization()(layer6)
    layer6 = layer6+layer2_
    layer6 = Dropout(0.5)(layer6)
    layer6 = ReLU()(layer6)

    layer7 = Conv2DTranspose(64, (3,3), strides=2, padding='valid')(layer6)
    layer7 = BatchNormalization()(layer7)
    layer7 = layer7+layer1_
    layer7 = Dropout(0.5)(layer7)
    layer7 = ReLU()(layer7)

    layer8 = Conv2DTranspose(3, (3,3), strides=2, padding='same')(layer7)

    outputs = layer8

    model = Model(inputs=inputs, outputs=outputs)

    return model

model = autoencoder()

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

modelpath = '../data/modelcheckpoint/noise_model.hdf5'
es = EarlyStopping(monitor='accuracy', patience=30)
cp = ModelCheckpoint(filepath=modelpath, monitor='accuracy', save_best_only=True)
lr = ReduceLROnPlateau(monitor='accuracy', patience=8, factor=0.8)
model.fit(x_train_noised, x_train, epochs=1000, callbacks=[es,cp,lr])

model = load_model('../data/modelcheckpoint/noise_model.hdf5')
output = model.predict(x_test_noised)


fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10), (ax11, ax12, ax13, ax14, ax15)) = plt.subplots(3, 5, figsize=(20,7))

# 이미지 5개를 무작위로 고른다.
random_imgs = random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다!!
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_imgs[i]].reshape(150,150,3))

    if i == 0:
        ax.set_ylabel('INPUT', size=20)

    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 잡음을 넣은 이미지
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[random_imgs[i]].reshape(150,150,3))

    if i == 0:
        ax.set_ylabel('Noise', size=20)

    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더가 출력한 이미지
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(output[random_imgs[i]].reshape(150,150,3))

    if i == 0:
        ax.set_ylabel('OUTPUT', size=20)

    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()
