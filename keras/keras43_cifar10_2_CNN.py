import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, x_test.shape)      # (50000, 32, 32, 3) (10000, 32, 32, 3)
print(y_train.shape, y_test.shape)      # (50000, 1) (10000, 1)

print(np.max(x_train), np.min(x_train))

x_train = x_train/255.
x_test = x_test/255.

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
model = Sequential()
model.add(Conv2D(filters=512, kernel_size=(2,2), padding='same', strides=1, input_shape=(32,32,3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv2D(256, (2,2), padding='same', strides=1))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=20, mode='auto')
model.fit(x_train, y_train, epochs=1000, batch_size=16, validation_split=0.2, verbose=2, callbacks=[es])

loss, acc = model.evaluate(x_test, y_test)
print('loss :', loss)
print('ACC :', acc)

# Result
# loss : 2.4829750061035156
# ACC : 0.5597000122070312


y_pred = model.predict(x_test)
print('==========================')
print('   예상 ' ,'|','   예측  ')
for i in range(10):
    print('    ', np.argmax(y_test[i+40]), '  |    ', np.argmax(y_pred[i+40]))