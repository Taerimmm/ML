# 실습
# 1. R2            : 0.5 이하 / 음수 X
# 2. layer         : 5개 이상
# 3. node          : 각 10개 이상
# 4. batch_size    : 8 이하
# 5. epochs        : 30 이상

import numpy as np

# 1. 데이터
x = np.array([range(100), range(301,401), range(1,101), range(100), range(301,401)])
y = np.array([range(711, 811), range(1,101)])
# print(x.shape)   # (5, 100) 
# print(y.shape)   # (2, 100)

x = np.transpose(x)
y = np.transpose(y)
# print(x.shape)   # (100, 5)
# print(y.shape)   # (100, 2)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=66)
# print(x_train.shape)   # (80, 5)
# print(y_train.shape)   # (80, 2)

# 2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# from keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_dim=5))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(2))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=30, batch_size=6, validation_split=0.2)

# 4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test)
print('loss :', loss)
print('mae :', mae)

y_predict = model.predict(x_test)
# print(y_predict)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print('RMSE :', RMSE(y_test, y_predict))
print('mse :', mean_squared_error(y_test, y_predict))

# R2 구하기
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('R2 :', r2)

# loss : 531.4566650390625
# mae : 19.325115203857422
# RMSE : 23.05334262662151
# mse : 531.4566062604044
# R2 : 0.32754888794783876