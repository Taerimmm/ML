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
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(1))
# model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto')

model.fit(x_train, y_train, epochs=2000, batch_size=8, validation_split=0.2, verbose=2, callbacks=[early_stopping])

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
# loss : 9.10645866394043
# MAE : 2.166841745376587
# RMSE : 3.0176911097836316
# MSE : 9.106459634067166
# R2 : 0.8906050141122304
