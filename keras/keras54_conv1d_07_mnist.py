# 실습
# Conv1d로 코딩
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)     # (60000, 28, 28), (60000,) <- 흑백
print(x_test.shape, y_test.shape)       # (10000, 28, 28), (10000,)

x_train = x_train.astype('float32')/255.
x_test = x_test/255. # (x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train)
print(y_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout

model = Sequential()
model.add(Conv1D(filters=256, kernel_size=2, padding='same',
                 strides=1, input_shape=(28,28)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv1D(64, 2, padding='same', strides=1))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv1D(64, 2, padding='same', strides=1))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=10, mode='auto')
model.fit(x_train, y_train, epochs=4000, batch_size=32, validation_split=0.2, verbose=2, callbacks=[es])

loss, acc = model.evaluate(x_test, y_test)
print('loss :', loss)
print('acc :', acc)

# Result
# loss : 0.05688846483826637
# acc : 0.9869999885559082

y_pred = model.predict(x_test)

print('==========================')
print('   예상 ' ,'|','   예측  ')
for i in range(10):
    print('  ', np.argmax(y_test[i+40]), '  |', np.argmax(y_pred[i+40]))
