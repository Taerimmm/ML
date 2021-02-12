import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.model_selection import KFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

train_datagen = ImageDataGenerator(
    rescale=1./255)
test_datagen = ImageDataGenerator(
    rescale=1./255)

''' ImgDatagen으로 50000개 image 로드 해보기 '''
train_generator = train_datagen.flow_from_directory(
    '../data', 
    classes=['dirty_mnist_2nd'],
    batch_size=50000, 
    target_size=(64, 64), 
    color_mode='grayscale',
    class_mode=None,
    shuffle=False)

test_generator = test_datagen.flow_from_directory(
    '../data', 
    classes=['test_dirty_mnist_2nd'],
    batch_size=5000, 
    target_size=(64, 64), 
    color_mode='grayscale',
    class_mode=None,
    shuffle=False)

for i in train_generator:
    x_train = i
    break
print(x_train.shape)

y_train = pd.read_csv('./dacon3/data/dirty_mnist_2nd_answer.csv', index_col=0, header=0)
print(y_train.shape)

steps = 5
kfold = KFold(n_splits=steps, random_state=42, shuffle=True)


# VGG 모델
inputs = Input(shape=x_train.shape[1:], dtype='float32', name='input')
 
layer = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.01))(inputs)
layer = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.01))(layer)
layer = MaxPooling2D((2,2))(layer)
 
layer = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.01))(layer)
layer = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.01))(layer)
layer = MaxPooling2D((2,2))(layer)
 
layer = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.01))(layer)
layer = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.01))(layer)
layer = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.01))(layer)
layer = MaxPooling2D((2,2))(layer)
 
layer = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.01))(layer)
layer = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.01))(layer)
layer = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.01))(layer)
layer = MaxPooling2D((2,2))(layer)
 
# layer = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.01))(layer)
# layer = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.01))(layer)
# layer = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.01))(layer)
# layer = MaxPooling2D((2,2))(layer)
 
layer = Flatten()(layer)
# layer = Dense(4096, kernel_initializer='he_normal')(layer)
layer = Dense(2048, kernel_initializer='he_normal')(layer)
layer = Dense(1024, kernel_initializer='he_normal')(layer)
outputs = Dense(26, activation='sigmoid')(layer)

model = Model(inputs=inputs, outputs=outputs)

model.summary()


# 훈련
i=0
filepath = './dacon3/data/vision_2_model_{}.hdf5'.format(i)
es = EarlyStopping(monitor='val_loss', patience=160, mode='auto')
cp = ModelCheckpoint(filepath=filepath, monitor='val_loss', save_best_only=True, mode='auto')
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=100)

model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=2e-5), metrics=['acc'])
history = model.fit(x_train, y_train, epochs=3, batch_size=256, validation_split=0.2, callbacks=[es,cp,lr])


# es = EarlyStopping(monitor='val_loss', patience=160, mode='auto')
# lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=100)

# for i, (train_idx, val_idx) in enumerate(kfold.split(x_train, y_train)):
#     x_train_, x_val_ = x_train[train_idx], x_train[val_idx]
#     y_train_, y_val_ = y_train.iloc[train_idx,:], y_train.iloc[val_idx,:]
    
#     filepath = './dacon3/data/vision_2_model_{}.hdf5'.format(i)
#     cp = ModelCheckpoint(filepath=filepath, monitor='val_loss', save_best_only=True, mode='auto')

#     model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=2e-5), metrics=['acc'])

#     history = model.fit(x_train_, y_train_, epochs=50000, validation_data=(x_val_, y_val_), callbacks=[es,cp,lr])

# Test
submission = pd.read_csv('./dacon3/data/sample_submission.csv', index_col=0, header=0)

result = 0
steps = 1
for i in range(steps):
    model = load_model('./dacon3/data/vision_2_model_{}.hdf5'.format(i))

    result += model.predict_generator(test_generator) / steps

print(result)


print(submission)
submission.to_csv('./dacon3/data/submission_vgg16.csv')
