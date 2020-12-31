# 실습 : 19_1, 2, 3, 4, 5, EarlyStopping 까지
# 총 6개의 파일을 완성하시오.

import numpy as np
from sklearn.datasets import load_diabetes

dataset = load_diabetes()
x = dataset.data
y = dataset.target

print(x[:5])
print(y[:10])
print(x.shape, y.shape)
print(np.max(x), np.min(x))
# print(dataset.feature_names)
# print(dataset.DESCR)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=45)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(10,)))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, verbose=2)

loss, mse = model.evaluate(x_test, y_test)
print("loss :", loss)
print('MSE :', mse)
y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score
def rmse(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE :', rmse(y_test, y_predict))
print('MSE :', mean_squared_error(y_test, y_predict))

print('R2 :', r2_score(y_test, y_predict))

# Result
# loss : 2461.3046875
# MSE : 39.835235595703125
# RMSE : 49.61153750383391
# MSE : 2461.3046534943187
# R2 : 0.5011856015264784