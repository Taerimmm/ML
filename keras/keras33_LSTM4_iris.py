# sklearn dataset
# LSTM으로 모델링
# Dense와 성능비교
# 다중 분류

# sklearn dataset
# LSTM으로 모델링
# Dense와 성능비교
# 이진 분류

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

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(4,1)))
model.add(Dense(20, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=2)

loss, acc = model.evaluate(x_test, y_test)
print('loss :', loss)
print('acc :', acc)

# loss : 0.1587691307067871
# acc : 0.9333333373069763

y_pred = model.predict(x_test[-5:-1])
# print(y_pred)
print(y_test[-5:-1])

# 결과치 나오게 코딩할것 # argmax
print(np.argmax(y_pred, axis=1))