import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2)
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2)
test_datagen = ImageDataGenerator(
    rescale=1./255)

''' ImgDatagen으로 50000개 image 로드 해보기 '''
# Found 40000 images belonging to 1 classes.
train_generator = train_datagen.flow_from_directory(
    '../data', 
    classes=['dirty_mnist_2nd'],
    batch_size=16, 
    target_size=(256, 256), 
    color_mode='grayscale',
    class_mode=None,
    subset='training')
val_generator = val_datagen.flow_from_directory(
    '../data/dirty_mnist_2nd', 
    batch_size=16, 
    target_size=(256, 256), 
    color_mode='grayscale',
    class_mode=None,
    subset='validation')

# VGG 모델
inputs = Input(shape=(256, 256, 1), dtype='float32', name='input')
 
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
 
layer = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.01))(layer)
layer = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.01))(layer)
layer = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.01))(layer)
layer = MaxPooling2D((2,2))(layer)
 
layer = Flatten()(layer)
layer = Dense(4096, kernel_initializer='he_normal')(layer)
layer = Dense(2048, kernel_initializer='he_normal')(layer)
layer = Dense(1024, kernel_initializer='he_normal')(layer)
outputs = Dense(26, activation='softmax')(layer)

model = Model(inputs=inputs, outputs=outputs)

model.summary()


# 훈련

filepath = './dacon3/data/vision_2_model_{}.hdf5'.format(i)
es = EarlyStopping(monitor='val_loss', patience=160, mode='auto')
cp = ModelCheckpoint(filepath=filepath, monitor='val_loss', save_best_only=True, mode='auto')
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=100)


model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=2e-5), metrics=['acc'])

history = model.fit_generator(train_generator, epochs=300, validation_data=val_generator, validation_steps=16, callbacks=[es,cp,lr])
