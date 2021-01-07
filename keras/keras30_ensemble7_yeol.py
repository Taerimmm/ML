# 행이 다른 앙상블 모델에 대해 공부 !!!

import numpy as np
from numpy import array
# 1. 데이터
x1 = array([[1,2],[2,3],[3,4],[4,5],
            [5,6],[6,7],[7,8],[8,9],
            [9,10],[10,11],
            [20,30],[30,40],[40,50]])
x2 = array([[10,20,30],[20,30,40],[30,40,50],[40,50,60],
            [50,60,70],[60,70,80],[70,80,90],[80,90,100],
            [90,100,110],[100,110,120],
            [2,3,4],[3,4,5],[4,5,6]])
y1 = array([[10,20,30],[20,30,40],[30,40,50],[40,50,60],
            [50,60,70],[60,70,80],[70,80,90],[80,90,100],
            [90,100,110],[100,110,120],
            [2,3,4],[3,4,5],[4,5,6]])
y2 = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x1_pred = array([55,65])     # -> (2,)
x2_pred = array([65,75,85])     # -> (3,)

# shape 변경
print(x1.shape)     # (13, 2)
print(x2.shape)     # (13, 3)
print(y1.shape)     # (13, 3)
print(y2.shape)     # (13,)

x1 = x1.reshape(x1.shape[0], x1.shape[1], 1) # -> (13, 2, 1)
x2 = x2.reshape(x2.shape[0], x2.shape[1], 1) # -> (13, 3, 1)

# 2. 모델
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Input
from tensorflow.keras.layers import concatenate

input1 = Input(shape=(2,1))
dense1 = LSTM(1024, activation='relu')(input1)
dense1 = Dense(512, activation='relu')(dense1)
dense1 = Dense(256, activation='relu')(dense1)
dense1 = Dense(256, activation='relu')(dense1)

input2 = Input(shape=(3,1))
dense2 = LSTM(1024, activation='relu')(input2)
dense2 = Dense(512, activation='relu')(dense2)
dense2 = Dense(256, activation='relu')(dense2)
dense2 = Dense(256, activation='relu')(dense2)

merge1 = concatenate([dense1, dense2])
dense = Dense(256)(merge1)
dense = Dense(128)(dense)

dense1 = Dense(64)(dense)
dense1 = Dense(64)(dense1)
output1 = Dense(3)(dense1)

dense2 = Dense(64)(dense)
dense2 = Dense(64)(dense2)
output2 = Dense(1)(dense2)

model = Model(inputs=[input1, input2], outputs=[output1, output2])

model.summary()

'''
# train 시키는 x의 형태
print(np.array([x1,x2]).shape) -> (2, 13, 3, 1)
# predict도 이와 동일한 구조를 형성
print(np.array([x1_pred.reshape(1,3,1),x2_pred.reshape(1,3,1)]).shape)  -> (2, 1, 3, 1)
'''

# 3. 컴파일, 룬련
model.compile(loss='mse', optimizer='adam')
model.fit([x1, x2], [y1, y2], epochs=1000, batch_size=4, verbose=2)
# model.fit([x1, x2], [y1, y2], epochs=1000, batch_size=4, verbose=2, validation_split=0.2)
# validation_Split이 일정 비율을 분리하여 검증데이터로 사용하게 된다.
# 이 경우는 데이터의 개수가 적기 떄문에 분리하게 되면 loss가 커질 수 밖에 없다

# 4. 평가, 예측
loss = model.evaluate([x1,x2], [y1, y2])
print('Total loss :', loss[0])
print('Model 1 loss :', loss[1])
print('Model 2 loss :', loss[2])
print('=======================')

x1_pred = array([55,65]).reshape(1,2,1)
x2_pred = array([65,75,85]).reshape(1,3,1)

# 85의 근사치
y_pred1, y_pred2 = model.predict([x1_pred, x2_pred])
print('y_pred1 :', y_pred1)
print('y_pred2 :', y_pred2)