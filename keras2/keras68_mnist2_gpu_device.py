import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
    except RuntimeError as e:
        print(e)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)     # (60000, 28, 28), (60000,) 
print(x_test.shape, y_test.shape)       # (10000, 28, 28), (10000,)

print(x_train[0])
print(y_train[0])
 
print(x_train[0].shape)         # (28, 28)

x_train = x_train.reshape(60000,28,28,1)/255.
x_test = x_test.reshape(10000,28,28,1)/255. 

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.regularizers import l1, l2, l1_l2

model = Sequential()
model.add(Conv2D(filters=256, kernel_size=(2,2), padding='same',
                 strides=1, input_shape=(28,28,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Dropout(0.2))
model.add(Conv2D(32, (2,2), kernel_initializer='he_normal')) # weight -> 정규분포
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(32, (2,2), kernel_regularizer=l1(l1=0.01)))
model.add(Dropout(0.2))

model.add(Conv2D(32, (2,2), strides=2))
model.add(MaxPooling2D(pool_size=2))

model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=5)
hist = model.fit(x_train, y_train, epochs=30, batch_size=32, validation_split=0.2, verbose=2, callbacks=[es])

result = model.evaluate(x_test, y_test)
print('loss :', result[0])
print('acc :', result[1])
