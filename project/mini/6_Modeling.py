import os
import numpy as np
import pandas as pd
import librosa
import librosa.display

from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import warnings
warnings.filterwarnings('ignore')

X = np.load('./project/mini/data/X.npy')
y = np.load('./project/mini/data/y.npy')

print(X.shape)  # (8973, 128, 660, 1)
print(y.shape)  # (8973, 15)

x_train, x_val , y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# 모델
def cnn_model(x):

    inputs = Input(shape=x.shape[1:])
    layer = Conv2D(64, (3,3), padding='same', activation='relu')(inputs)
    layer = MaxPooling2D(pool_size=2)(layer)
    layer = Conv2D(128, (3,3), padding='same', activation='relu')(layer)
    layer = MaxPooling2D(pool_size=2)(layer)
    layer = Dropout(0.2)(layer)
    layer = Conv2D(256, (3,3), padding='same', activation='relu')(layer)
    layer = Conv2D(256, (3,3), padding='same', activation='relu')(layer)
    layer = MaxPooling2D(pool_size=2)(layer)
    layer = Conv2D(512, (3,3), padding='same', activation='relu')(layer)
    layer = Conv2D(512, (3,3), padding='same', activation='relu')(layer)
    layer = MaxPooling2D(pool_size=2)(layer)
    layer = Conv2D(512, (3,3), padding='same', activation='relu')(layer)
    layer = Conv2D(512, (3,3), padding='same', activation='relu')(layer)
    layer = MaxPooling2D(pool_size=2)(layer)
    layer = Dropout(0.2)(layer)    
    layer = Flatten()(layer)
    # layer = Dense(4096)(layer)
    layer = Dense(1024)(layer)
    layer = Dense(256)(layer)
    outputs = Dense(15, activation='softmax')(layer)

    model = Model(inputs=inputs, outputs=outputs)

    return model

model = cnn_model(X)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
file_path = './project/mini/data/genre_model.hdf5'
es = EarlyStopping(monitor='val_loss', patience=120)
cp = ModelCheckpoint(filepath=file_path, monitor='val_loss', save_best_only=True)
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=30)
history = model.fit(x_train, y_train, epochs=5000, batch_size=32, validation_data=(x_val, y_val), verbose=2, callbacks=[es,cp,lr])
