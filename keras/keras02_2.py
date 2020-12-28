import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
# from tensorflow.keras import models
# from tensorflow import keras
from tensorflow.keras.layers import Dense

# 1. 데이터
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([2,4,6,8,10,12,14,16,18,20])

x_test = np.array([101,102,103, 104, 105, 106, 107, 108, 109, 110])
y_test = np.array([111,112,113, 114, 115, 116, 117, 118, 119, 120])
# y_test = np.array([202,204,206, 208, 210, 212, 214, 216, 218, 220])

x_predict = np.array([111,112,113])

# 2. 모델
model = Sequential()
# model = models.Sequential()
# model = keras.models.Sequential()
model.add(Dense(5, input_dim = 1, activation='linear'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=200, batch_size=2)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=2)
print('loss :', loss)

result = model.predict(x_predict)
print('result :', result)

