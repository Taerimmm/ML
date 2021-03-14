import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display

from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.layers import BatchNormalization, ZeroPadding2D, Activation, Add, GlobalAveragePooling2D
from tensorflow.keras.applications import resnet
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import warnings
warnings.filterwarnings('ignore')

X = np.load('./project/mini/data/X.npy')
y = np.load('./project/mini/data/y.npy')

print(X.shape)  # (6194, 128, 660, 1)
print(y.shape)  # (6194, 13)

x_train, x_val , y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# MinMax
print(np.min(X), np.max(X))

x_train /= -80
x_val /= -80

# 모델
input_tensor = Input(shape=X.shape[1:], dtype='float32', name='input')

model = resnet.ResNet101(input_tensor=input_tensor, weights=None, pooling='max', classes=13)
 
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
file_path = './project/mini/data/genre_model_resnet50_app.hdf5'
es = EarlyStopping(monitor='val_accuracy', patience=50)
cp = ModelCheckpoint(filepath=file_path, monitor='val_accuracy', save_best_only=True)
lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=20)
history = model.fit(x_train, y_train, epochs=5000, batch_size=32, validation_data=(x_val, y_val), verbose=2, callbacks=[es,cp,lr])

# Epoch 492/5000
# 155/155 - 47s - loss: 4.0296e-05 - accuracy: 1.0000 - val_loss: 4.2372 - val_accuracy: 0.6521  
# 아마 이때 cp가 저장된 것 같음.


# history plot 해서 그리기

# plt
train_loss = history.history['accuracy']
test_loss = history.history['val_accuracy']

plt.figure(figsize=(12,8))
plt.plot(train_loss, label='Training accuracy', color='blue')
plt.plot(test_loss, label='Testing accuracy', color='red')

plt.title('Training and Testing Accuracy by Epoch', fontsize=25)
plt.xlabel('Epoch', fontsize=18)
plt.ylabel('Accuracy', fontsize=18)
# plt.xticks(range(1,len(train_loss), range(1,len(test_loss))))
plt.legend()
plt.show()

