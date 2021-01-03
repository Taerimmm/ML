# 2개의 파일을 만드시오.
# 1. EarlyStopping 을 적용해서 않은 최고의 모델
# 2. EarlyStopping 을 적용한 최고의 모델

#주말 과제 #
import numpy as np

from tensorflow.keras.datasets import boston_housing

boston = boston_housing.load_data()
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

print(x_train.shape)
print(y_train.shape)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(13,)))
model.add(Dense(256))
model.add(Dense(256))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(1))
# model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=200, batch_size=4, validation_split=0.2, verbose=0)

loss, mae = model.evaluate(x_test, y_test)
print('loss :', loss)
print('MAE :', mae, '\n')

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score
def rmse(y_test, y_predict):
  return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE :', rmse(y_test, y_predict))
print('MSE :', mean_squared_error(y_test, y_predict))

print('R2 :', r2_score(y_test, y_predict))


# 결과
# loss : 10.722761154174805
# MAE : 2.279066801071167 
# RMSE : 3.27456299498727
# MSE : 10.722762808139999
# R2 : 0.8711885262538106
