import numpy as np
import pandas as pd

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

X = np.load('./project/mini/data/X.npy')
y = np.load('./project/mini/data/y.npy')

print(X.shape)  # (6194, 128, 660, 1)
print(y.shape)  # (6194, 13)

x_train, x_val , y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

x_train /= -80
x_val /= -80

# 모델
model = Sequential()
model.add(Flatten(input_shape=(X.shape[1:])))
model.add(Dense(1024, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(13, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
file_path = './project/mini/data/genre_model.hdf5'
es = EarlyStopping(monitor='val_loss', patience=120)
cp = ModelCheckpoint(filepath=file_path, monitor='val_loss', save_best_only=True)
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=30)
history = model.fit(x_train, y_train, epochs=5000, batch_size=32, validation_data=(x_val, y_val), verbose=2, callbacks=[es,cp,lr])
