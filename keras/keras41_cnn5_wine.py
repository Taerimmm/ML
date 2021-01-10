# CNN으로 구성
# 2차원을 4차원으로 늘여서 하시오.

import numpy as np
from sklearn.datasets import load_wine

dataset = load_wine()
x = dataset.data
y = dataset.target

print(x.shape)    # (178, 13)
print(y.shape)    # (178,)

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y.shape)    # (178, 3)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=45)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1, 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1, 1)

# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
model = Sequential()
model.add(Conv2D(filters=256, kernel_size=(2,2), padding='same', strides=1, input_shape=(13,1,1)))
model.add(Dropout(0.2))
model.add(Conv2D(128, (2,2), padding='same', strides=1))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=200, batch_size=4, verbose=2)

loss, acc = model.evaluate(x_test, y_test)
print('loss :', loss)
print('acc :', acc)

# loss : 0.06733787804841995
# acc : 0.9722222089767456

y_pred = model.predict(x_test[-5:-1])
# print(y_pred)
print(y_test[-5:-1])

# 결과치 나오게 코딩할것 # argmax
print(np.argmax(y_pred, axis=1))
