import os
import numpy as np
import pandas as pd
import tensorflow as tf
import librosa
import librosa.display

from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, MaxPooling2D, Dropout, GRU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import warnings
warnings.filterwarnings('ignore')

X = np.load('./project/mini/data/X.npy')
y = np.load('./project/mini/data/y.npy')

X = X.reshape(X.shape[:-1])

print(X.shape)  # (6194, 128, 660, 1)
print(y.shape)  # (6194, 13)

x_train, x_val , y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

x_train /= -80
x_val /= -80

# 모델
def rnn_model(x):
    model = Sequential()
    model.add(GRU(64, activation='relu', input_shape=x.shape[1:])) # (3, 1) -> (timesteps, input_dim)
    model.add(Dense(64))
    model.add(Dense(128))
    model.add(Dense(256))
    model.add(Dense(512))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(13, activation='softmax'))

    return model

model = rnn_model(x_train)

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
file_path = './project/mini/data/genre_model_gru.hdf5'
es = EarlyStopping(monitor='val_accuracy', patience=80)
cp = ModelCheckpoint(filepath=file_path, monitor='val_accuracy', save_best_only=True)
lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.8, patience=30)
history = model.fit(x_train, y_train, epochs=5000, batch_size=32, validation_data=(x_val, y_val), verbose=2, callbacks=[es,cp,lr])
 