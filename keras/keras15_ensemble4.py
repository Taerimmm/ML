# 실습 1:다 앙상블 구현

import numpy as np

# 1. 데이터
x1 = np.array([range(100), range(301,401), range(1,101)])

y1 = np.array([range(711, 811), range(1,101), range(201, 301)])
y2 = np.array([range(501,601), range(711,811), range(100)])

x1 = np.transpose(x1)

y1 = np.transpose(y1)
y2 = np.transpose(y2)

from sklearn.model_selection import train_test_split
x1_train, x1_test = train_test_split(x1, train_size=0.8, shuffle=False)
y1_train, y1_test, y2_train, y2_test = train_test_split(y1, y2, train_size=0.8, shuffle=False)

# 2. 모델 구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

# 모델 1
input1 = Input(shape=(3,))  
dense1 = Dense(10, activation='relu')(input1)
dense1 = Dense(5, activation='relu')(dense1)
dense1 = Dense(5, activation='relu')(dense1)
dense1 = Dense(5, activation='relu')(dense1)
# output1 = Dense(3)(dense1)

# 모델 분기 1 - 8 layer
output1 = Dense(30)(dense1)
output1 = Dense(10)(output1)
output1 = Dense(70)(output1)
output1 = Dense(10)(output1)
output1 = Dense(50)(output1)
output1 = Dense(10)(output1)
output1 = Dense(20)(output1)
output1 = Dense(3)(output1)

# 모델 분기 2 - 8 layer
output2 = Dense(15)(dense1)
output2 = Dense(10)(output2)
output2 = Dense(70)(output2)
output2 = Dense(10)(output2)
output2 = Dense(20)(output2)
output2 = Dense(10)(output2)
output2 = Dense(70)(output2)
output2 = Dense(3)(output2)

# 모델 선언
model = Model(inputs=input1, outputs=[output1, output2])
model.summary()

# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x1_train, [y1_train, y2_train], epochs=100, batch_size=1, validation_split=0.2, verbose=0)

# 4. 평가, 예측
loss = model.evaluate(x1_test, [y1_test, y2_test])
print('Total loss :', loss[0])
print('Model 1 loss :', loss[1])
print('Model 2 loss :', loss[2])

print('model.metrics_names :', model.metrics_names)
print(loss,'\n')

y1_predict, y2_predict = model.predict(x1_test)
print("====================================")
print('y1_predict : \n', y1_predict)
print("====================================")
print('y2_predict : \n', y2_predict)
print("====================================")

# RMSE
from sklearn.metrics import mean_squared_error
def rmse(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

RMSE1 = rmse(y1_test, y1_predict)
RMSE2 = rmse(y2_test, y2_predict)
RMSE = (RMSE1 + RMSE2) / 2

print('RMSE 1 :', RMSE1)
print('MSE 1 :', mean_squared_error(y1_test, y1_predict))
print('RMSE 2 :', RMSE2)
print('MSE 2 :', mean_squared_error(y2_test, y2_predict))
print('RMSE :', RMSE,'\n')

# R2
from sklearn.metrics import r2_score
r2_1 =  r2_score(y1_test, y1_predict)
r2_2 =  r2_score(y2_test, y2_predict)
r2 = (r2_1 + r2_2) / 2

print('R2 1 :', r2_1)
print('R2 2 :', r2_2)
print('R2:', r2)


x_predict = np.array([100,400,100]).reshape(1,3)
y_predict1, y_predict2 = model.predict(x_predict)
print(y_predict1, y_predict2)