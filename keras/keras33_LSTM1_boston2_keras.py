# tensorflow dataset
# LSTM으로 모델링
# Dense와 성능비교
# 회귀모델

import numpy as np

# 1. 데이터
from tensorflow.keras.datasets import boston_housing
dataset = boston_housing.load_data()
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

print(x_train.shape)    # (404, 13)
print(x_test.shape)     # (102, 13)
print(y_train.shape)    # (404,)
print(y_test.shape)     # (102,)

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
model.add(LSTM(10, activation='relu', input_shape=(13,1)))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=2)

loss = model.evaluate(x_test, y_test)
print('loss :', loss)

y_pred = model.predict(x_test)
from sklearn.metrics import mean_squared_error, r2_score
def rmse(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))
print('RMSE :', rmse(y_test, y_pred))
print('MSE :', mean_squared_error(y_test, y_pred))

print('R2 :', r2_score(y_test, y_pred))

# LSTM
# loss : 29.691123962402344
# RMSE : 5.448956350273748
# MSE : 29.691125307188603
# R2 : 0.6433234907426657