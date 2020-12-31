# metrics가 다른 경우

import numpy as np

# 1. 데이터
x1 = np.array([range(100), range(301,401), range(1,101)])
y1 = np.array([range(711, 811), range(1,101), range(201, 301)])

x2 = np.array([range(101,201), range(411,511), range(100,200)])
y2 = np.array([range(501,601), range(711,811), range(100)])

x1 = np.transpose(x1)
x2 = np.transpose(x2)
y1 = np.transpose(y1)
y2 = np.transpose(y2)

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, train_size=0.8, shuffle=False)
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, train_size=0.8, shuffle=False)

# 2. 모델 구성

# 우선 함수형으로 모델을 2개 만든 후 합친다
# Sequential은 layer를 쌓는 느낌이라 가중치가 섞이기 떄문에 잘 사용안한다고 함

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

# 모델 1
input1 = Input(shape=(3,))  
dense1 = Dense(10, activation='relu')(input1)
dense1 = Dense(5, activation='relu')(dense1)
# output1 = Dense(3)(dense1)

# 모델 2
input2 = Input(shape=(3,))
dense2 = Dense(10, activation='relu')(input2)
dense2 = Dense(5, activation='relu')(dense2)
dense2 = Dense(5, activation='relu')(dense2)
dense2 = Dense(5, activation='relu')(dense2)
# output2 = Dense(3)(dense2)

# 모델 병합 / concatenate
from tensorflow.keras.layers import concatenate, Concatenate
# from keras.layers.merge import concatenate, Concatenate
# from keras.layers import concatenate, Concatenate
merge1 = concatenate([dense1, dense2])
middle1 = Dense(30)(merge1)
middle1 = Dense(10)(middle1)
middle1 = Dense(10)(middle1)

# 모델 분기 1
output1 = Dense(30)(middle1)
output1 = Dense(7)(output1)
output1 = Dense(3)(output1)

# 모델 분기 2
output2 = Dense(15)(middle1)
output2 = Dense(7)(output2)
output2 = Dense(7)(output2)
output2 = Dense(3)(output2)

# 모델 선언
model = Model(inputs=[input1, input2], outputs=[output1, output2])
model.summary()

# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae', 'accuracy', 'hinge'])
model.fit([x1_train, x2_train], [y1_train, y2_train], epochs=10, batch_size=1, validation_split=0.2, verbose=1)

# 4. 평가, 예측
loss = model.evaluate([x1_test, x2_test], [y1_test, y2_test])
print('Total loss :', loss[0])
print('Model 1 loss :', loss[1])
print('Model 2 loss :', loss[2])

print('model.metrics_names :', model.metrics_names)
print(loss,'\n')

y1_predict, y2_predict = model.predict([x1_test, x2_test])
# print("====================================")
# print('y1_predict : \n', y1_predict)
# print("====================================")
# print('y2_predict : \n', y2_predict)
# print("====================================")

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