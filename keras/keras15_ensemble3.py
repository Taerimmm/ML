# 실습 다:다 앙상블을 구현하시오.

import numpy as np

# 1. 데이터
x1 = np.array([range(100), range(301,401), range(1,101)])
x2 = np.array([range(101,201), range(411,511), range(100,200)])

y1 = np.array([range(711, 811), range(1,101), range(201, 301)])
y2 = np.array([range(501,601), range(711,811), range(100)])
y3 = np.array([range(601,701), range(811,911), range(1100,1200)])

x1 = np.transpose(x1)
x2 = np.transpose(x2)
y1 = np.transpose(y1)
y2 = np.transpose(y2)
y3 = np.transpose(y3)

from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test = train_test_split(x1, x2, train_size=0.8, shuffle=False)
y1_train, y1_test, y2_train, y2_test, y3_train, y3_test = train_test_split(y1, y2, y3, train_size=0.8, shuffle=False)

# 2. 모델 구성

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

# 모델 분기 1 - 5 layer
output1 = Dense(30)(middle1)
output1 = Dense(10)(output1)
output1 = Dense(7)(output1)
output1 = Dense(10)(output1) 
output1 = Dense(3)(output1)

# 모델 분기 2 - 6 layer
output2 = Dense(15)(middle1)
output2 = Dense(7)(output2)
output2 = Dense(10)(output2)
output2 = Dense(7)(output2)
output2 = Dense(10)(output2)
output2 = Dense(3)(output2)

# 모델 분기 3 - 8 layer
output3 = Dense(15)(middle1)
output3 = Dense(7)(output3)
output3 = Dense(10)(output3)
output3 = Dense(4)(output3)
output3 = Dense(10)(output3)
output3 = Dense(7)(output3)
output3 = Dense(5)(output3)
output3 = Dense(3)(output3)

# 모델 선언
model = Model(inputs=[input1, input2], outputs=[output1, output2, output3])
model.summary()

# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit([x1_train, x2_train], [y1_train, y2_train, y3_train], epochs=100, batch_size=1, validation_split=0.2, verbose=2)

# 4. 평가, 예측
loss = model.evaluate([x1_test, x2_test], [y1_test, y2_test, y3_test])
print('Total loss :', loss[0])
print('Model 1 loss :', loss[1])
print('Model 2 loss :', loss[2])
print('Model 3 loss :', loss[3])

print('model.metrics_names :', model.metrics_names)
print(loss,'\n')

y1_predict, y2_predict, y3_predict = model.predict([x1_test, x2_test])
print("====================================")
print('y1_predict : \n', y1_predict)
print("====================================")
print('y2_predict : \n', y2_predict)
print("====================================")
print('y3_predict : \n', y3_predict)
print("====================================")

# RMSE
from sklearn.metrics import mean_squared_error
def rmse(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

RMSE1 = rmse(y1_test, y1_predict)
RMSE2 = rmse(y2_test, y2_predict)
RMSE3 = rmse(y3_test, y3_predict)
RMSE = (RMSE1 + RMSE2 + RMSE3) / 3

print('RMSE 1 :', RMSE1)
print('MSE 1 :', mean_squared_error(y1_test, y1_predict))
print('RMSE 2 :', RMSE2)
print('MSE 2 :', mean_squared_error(y2_test, y2_predict))
print('RMSE 3 :', RMSE3)
print('MSE 3 :', mean_squared_error(y3_test, y3_predict))
print('RMSE :', RMSE,'\n')

# RMSE 1 : 0.22985970161144206
# MSE 1 : 0.05283548242490119
# RMSE 2 : 0.2492566623996333
# MSE 2 : 0.06212888375060477
# RMSE 3 : 0.4817468258744169
# MSE 3 : 0.23208000424007577
# RMSE : 0.32028772996183075

# R2
from sklearn.metrics import r2_score
r2_1 =  r2_score(y1_test, y1_predict)
r2_2 =  r2_score(y2_test, y2_predict)
r2_3 =  r2_score(y3_test, y3_predict)
r2 = (r2_1 + r2_2 + r2_3) / 3

print('R2 1 :', r2_1)
print('R2 2 :', r2_2)
print('R2 3 :', r2_3)
print('R2:', r2)

# R2 1 : 0.9984109629345893
# R2 2 : 0.998131462142839
# R2 3 : 0.9930201502484187
# R2: 0.9965208584419489


x_predict = np.array([[100,400,100],[200,500,200]])
print(x_predict.shape)
y_predict = model.predict(x_predict)
print(y_predict)


x_predict1 = np.array([100,400,100]).reshape(1,3)
x_predict2 = np.array([200,500,200]).reshape(1,3)
y_predict1, y_predict2, y_predict3 = model.predict([x_predict1, x_predict2])
print(y_predict1, y_predict2, y_predict3)