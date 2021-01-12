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

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

model = Sequential()
model.add(Conv2D(filters=256, kernel_size=(2,2), padding='same',
                 strides=1, input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv2D(64, (2,2), padding='same', strides=1))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv2D(64, (2,2), padding='same', strides=1))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

model.save('../data/h5/k52_1_model1.h5')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# k52_1_minst_??? => k52_1_MCK_0.0367.hdf5 이름을 바꿔줄 것
modelpath = '../data/modelcheckpoint/k52_1_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
es = EarlyStopping(monitor='val_loss', patience=5)
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
hist = model.fit(x_train, y_train, epochs=30, batch_size=32, validation_split=0.2, verbose=2, callbacks=[es,cp]) 

model.save('../data/h5/k52_1_model2.h5')
model.save_weights('../data/h5/k52_1_weight.h5')

result = model.evaluate(x_test, y_test)
print('loss :', result[0])
print('accuracy :', result[1])
