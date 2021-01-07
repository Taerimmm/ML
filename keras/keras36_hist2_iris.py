# hist를 이용하여 그래프를 그리시오.
# loss, val_loss, acc, val_acc

import numpy as np

# 1. 데이터
from sklearn.datasets import load_iris
dataset = load_iris()
x = dataset.data
y = dataset.target

print(x.shape)      # (150, 4)
print(y.shape)      # (150,)

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y.shape)      # (150, 3)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=45)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(4,)))
model.add(Dense(20, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, verbose=2)
print(hist.history.keys())

import matplotlib.pyplot as plt
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])

plt.title('loss & acc')
plt.xlabel('loss, acc')
plt.ylabel('epoch')
plt.legend(['train loss', 'val loss', 'train acc', 'val acc'])
plt.show()