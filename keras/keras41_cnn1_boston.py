# CNN으로 구성
# 2차원을 4차원으로 늘여서 하시오.

import numpy as np

# 1. 데이터
from sklearn.datasets import load_boston
dataset = load_boston()
x = dataset.data
y = dataset.target
print(x.shape)      # (506, 13)
print(y.shape)      # (506,)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=45)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape)
print(x_test.shape)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1, 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1, 1)

# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
model = Sequential()
model.add(Conv2D(filters=256, kernel_size=(2,2), padding='same', strides=1, input_shape=(13,1,1)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (2,2), padding='same', strides=1))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
# from tensorflow.keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='loss', patience=20, mode='auto')
model.fit(x_train, y_train, epochs=400, batch_size=8, verbose=2)

loss = model.evaluate(x_test, y_test)
print('loss :', loss)

y_pred = model.predict(x_test)
from sklearn.metrics import mean_squared_error, r2_score
def rmse(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))
print('RMSE :', rmse(y_test, y_pred))
print('MSE :', mean_squared_error(y_test, y_pred))
print('R2 :', r2_score(y_test, y_pred))

# CNN
# loss : 8.528459548950195
# RMSE : 2.920352573671342
# MSE : 8.52845915454883
# R2 : 0.9210897579367158

# LSTM
# loss : 17.280284881591797
# RMSE : 4.156956092360923
# MSE : 17.28028395381659
# R2 : 0.8401128076001099

# Dense
# loss : 7.321578502655029
# MAE : 2.055612564086914
# RMSE : 2.7058414951171224
# R2 : 0.9322565193410672
