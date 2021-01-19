import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)     # (60000, 28, 28), (60000,) 
print(x_test.shape, y_test.shape)       # (10000, 28, 28), (10000,)

print(x_train[0])
print(y_train[0])

print(x_train[0].shape)         # (28, 28)

x_train = x_train.reshape(60000,28,28,1)/255.
x_test = x_test.reshape(10000,28,28,1)/255. 

y_train = x_train
y_test = x_test

print(y_train.shape)        # (60000, 28, 28, 1)
print(y_test.shape)         # (10000, 28, 28, 1)

# from tensorflow.keras.utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

model = Sequential()
# model.add(Conv2D(filters=64, kernel_size=(2,2), padding='same', strides=1, input_shape=(28,28,1)))
# model.add(MaxPooling2D(pool_size=2))
model.add(Dense(64, input_shape=(28,28,1)))
model.add(Dropout(0.5))
# model.add(Conv2D(1, (2,2)))
# model.add(Conv2D(1, (2,2)))
model.add(Dense(64))
# model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1))
'''
4차원 -> 4차원
'''

model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
modelpath = '../data/modelcheckpoint/k57_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
es = EarlyStopping(monitor='val_loss', patience=10)
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)
hist = model.fit(x_train, y_train, epochs=50, batch_size=16, validation_split=0.5, verbose=2, callbacks=[es,cp,reduce_lr])

result = model.evaluate(x_test, y_test)
print('loss :', result[0])
print('acc :', result[1])

y_pred = model.predict(x_test)
print(y_pred[0])
print(y_pred.shape)