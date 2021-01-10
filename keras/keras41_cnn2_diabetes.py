# CNN으로 구성
# 2차원을 4차원으로 늘여서 하시오.

import numpy as np

# 1. 데이터
from sklearn.datasets import load_diabetes
dataset = load_diabetes()
x = dataset.data
y = dataset.target

print(x.shape)      # (442, 10)
print(y.shape)      # (442,)

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
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(2,2), padding='same', strides=1, input_shape=(10,1,1)))
model.add(Dropout(0.2))
model.add(Conv2D(32, (2,2), padding='same', strides=1))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=2)

loss = model.evaluate(x_test, y_test)
print('loss :', loss)

y_pred = model.predict(x_test)
from sklearn.metrics import mean_squared_error, r2_score
def rmse(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE :", rmse(y_test, y_pred))
print('MSE :', mean_squared_error(y_test, y_pred))

print('R2 :', r2_score(y_test, y_pred))

# CNN
# loss : 2514.091552734375
# RMSE : 50.14071907425272
# MSE : 2514.091709283131
# R2 : 0.49048763959677477

# LSTM
# loss : 3504.971923828125
# RMSE : 59.20280262208613
# MSE : 3504.971838309689
# R2 : 0.2896732971633672
