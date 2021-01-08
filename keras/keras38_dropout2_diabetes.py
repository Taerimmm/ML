# 실습
# Dropout 적용

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
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=.2, random_state=45)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(10,)))
model.add(Dropout(0.2))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_data=(x_val, y_val), verbose=2)

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

# Dropout 이전
# loss : 2524.6904296875
# MSE : 40.10665512084961
# RMSE : 50.246297412319414
# MSE : 2524.690403647257
# R2 : 0.4883396806489295

# Dropout 후
# loss : 3258.488037109375
# MSE : 44.7831916809082
# RMSE : 57.08316644392718
# MSE : 3258.4878912650943
# R2 : 0.3396263745298267