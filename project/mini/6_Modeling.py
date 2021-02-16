import os
import numpy as np
import pandas as pd
import librosa
import librosa.display

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


data = pd.read_csv('./project/mini/data/total_genres_mfcc.csv', header=0)

label_dict = {
    'hiphop':0,
    'rock':1,
    'pop':2,
    'instrumental':3,
    'folk':4,
    'electronic':5,
    'international':6,
    'experimental':7,
    'jazz':8,
    'blues':9,
    'classical':10,
    'reggae':11,
    'disco':12,
    'metal':13,
    'country':14
}

y = data.iloc[:,-1].map(label_dict).values
print(y)
print(type(y))

y = to_categorical(y)
print(y)
print(type(y))

X = data.iloc[:,1:-1]
print(X)

steps = 5
kfold = KFold(n_splits=steps, shuffle=True)

# 모델
def cnn_model(x):

    inputs = Input(shape=x.shape[1:])
    layer = Dense(32,activation='relu')(inputs)
    outputs = Dense(15, activation='softmax')(layer)

    model = Model(inputs=inputs, outputs=outputs)

    return model

model = cnn_model(X)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
file_path = './project/mini/data/genre_model.hdf5'
es = EarlyStopping(monitor='val_loss', patience=40)
cp = ModelCheckpoint(filepath=file_path, monitor='val_loss', save_best_only=True)
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=20)
history = model.fit(X, y, epochs=1000, batch_size=32, validation_split=0.2, verbose=2)
