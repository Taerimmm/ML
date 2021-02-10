# 내 사진 predict
# 남자 acc=0.99 식으로 출력할 것
# predict 부분과 결과치는 메일로 보낼 것

import numpy as np
import matplotlib.pyplot as plt
import cv2

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

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
    fill_mode='nearest',
    validation_split=0.2
)
test_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2) 

# male - 841 (1) , female - 895 (0)
xy_train = train_datagen.flow_from_directory(
    '../data/image/human',
    target_size=(150,150),
    batch_size=20,
    class_mode='binary',
    shuffle=True,
    subset='training'
)

xy_val = train_datagen.flow_from_directory(
    '../data/image/human',
    target_size=(150,150),
    batch_size=20,
    class_mode='binary',
    shuffle=False,
    subset='validation'
)

print(xy_train)
print(xy_val)

# 2. 모델
model = Sequential()
model.add(Conv2D(36, (3,3), padding='same', activation='relu', input_shape=(150,150,3)))
model.add(BatchNormalization())
model.add(Conv2D(36, (3,3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=3))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3,3), padding='same', activation='relu', input_shape=(150,150,3)))
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=3))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', patience=20, mode='auto')
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=7, mode='auto')
history = model.fit_generator(xy_train, epochs=200, validation_data=xy_val, validation_steps=4, callbacks=[es, lr])

model.save('../data/h5/k67_img.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

print('acc :', acc[-1])
print('val_acc :', val_acc[-1])

# acc : 0.6184305548667908
# val_acc : 0.550000011920929

# My_Picture Predict
from tensorflow.keras.models import load_model
model = load_model('../data/h5/k67_img.h5')

pred_datagen = ImageDataGenerator(rescale=1./255) 

pred_data = pred_datagen.flow_from_directory(
    '../data/image',
    classes=['my'],
    target_size=(150,150),
    batch_size=1,
    class_mode=None
)
print(pred_data)

pred = model.predict_generator(pred_data)
print(pred)

print('======================================')
if pred > 0.5:
    print("남자 acc =", pred)
else:
    print("여자 acc =", pred)

# 남자 acc = [[0.6553112]]

img = cv2.imread('../data/image/my/my.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, dsize=(150,150)) / 255.0

result = model.predict(np.array([img]))
print(result)

print('======================================')
if result > 0.5:
    print("남자 acc =", result)
else:
    print("여자 acc =", result)

# 남자 acc = [[0.6553112]]
