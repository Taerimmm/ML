# 실습
# 남자 여자 구별
# ImageDataGenerator / fit 사용해서 완성    

import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

# 1. 데이터
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1./255) 

# male - 841 (1) , female - 895 (0)

# train_test_split()

xy_train = train_datagen.flow_from_directory(
    '../data/image/human',
    target_size=(150,150),
    batch_size=500,
    class_mode='binary',
)

xy_test = test_datagen.flow_from_directory(
    '../data/image/human',
    target_size=(150,150),
    batch_size=100,
    class_mode='binary',
)

# 2. 모델
model = Sequential()
model.add(Conv2D(256, (3,3), padding='same', activation='relu', input_shape=(150,150,3)))
model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=3))
model.add(Dropout(0.2))
model.add(Conv2D(512, (3,3), padding='same', activation='relu'))
model.add(Conv2D(512, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=3))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

model.fit(xy_train[0][0], xy_train[0][1], epochs=100, batch_size=32, validation_split=0.2, verbose=2)