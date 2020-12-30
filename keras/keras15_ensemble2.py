# 실습 다:1 앙상블 구현

import numpy as np

# 1. 데이터
x1 = np.array([range(100), range(301,401), range(1,101)])
x2 = np.array([range(101,201), range(411,511), range(100,200)])

y1 = np.array([range(711, 811), range(1,101), range(201, 301)])
# y2 = np.array([range(501,601), range(711,811), range(100)])

x1 = np.transpose(x1)
x2 = np.transpose(x2)
y1 = np.transpose(y1)
# y2 = np.transpose(y2)

from sklearn.model_selection import train_test_split
# x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, train_size=0.8, shuffle=False)
# x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, train_size=0.8, shuffle=False)

x1_train, x1_test, x2_train, x2_test, y1_train, y1_test = train_test_split(x1, x2, y1, train_size=0.8, shuffle=False)

print(x1_train.shape)
print(x1_test.shape)
print(x2_train.shape)
print(x2_test.shape)
print(y1_train.shape)
print(y1_test.shape)

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
output1 = Dense(30)(merge1)
output1 = Dense(10)(output1)
output1 = Dense(3)(output1)

# 모델 선언
model = Model(inputs=[input1, input2], outputs=output1)
model.summary()

# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit([x1_train, x2_train], y1_train, epochs=10, batch_size=1, validation_split=0.2, verbose=1)

# 4. 평가, 예측
loss = model.evaluate([x1_test, x2_test], y1_test)
print('loss :', loss[0])
print('MAE :', loss[1])

print('model.metrics_names :', model.metrics_names)
print(loss,'\n')

y1_predict = model.predict([x1_test, x2_test])
print("====================================")
print('y1_predict : \n', y1_predict)
print("====================================",'\n')


# RMSE
from sklearn.metrics import mean_squared_error
def rmse(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print('RMSE :', rmse(y1_test, y1_predict))
print('MSE :', mean_squared_error(y1_test, y1_predict), '\n')

# R2
from sklearn.metrics import r2_score
r2 = r2_score(y1_test, y1_predict)
print('R2:', r2)