import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, x_test.shape)      # (50000, 32, 32, 3) (10000, 32, 32, 3)
print(y_train.shape, y_test.shape)      # (50000, 1) (10000, 1)

print(np.max(x_train), np.min(x_train))

x_train = x_train.reshape(50000,32*32*3)/255.
x_test = x_test.reshape(10000,32*32*3)/255.

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
model = Sequential()
model.add(Dense(1024, activation='relu', input_shape=(32*32*3,)))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=20, mode='auto')
model.fit(x_train, y_train, epochs=4000, batch_size=32, validation_split=0.2, verbose=2, callbacks=[es])

loss, acc = model.evaluate(x_test, y_test)
print('loss :', loss)
print('ACC :', acc)

# Result
# loss : 1.534844160079956
# ACC : 0.47429999709129333

y_pred = model.predict(x_test)
print('==========================')
print('   예상 ' ,'|','   예측  ')
for i in range(10):
    print('    ', np.argmax(y_test[i+40]), '  |    ', np.argmax(y_pred[i+40]))