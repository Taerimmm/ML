# 2개의 모델을 하나는 LSTM, 하나는 Dense로
# 앙상블 구한다
# 29_1 번과 성능비교

import numpy as np
from numpy import array
# 1. 데이터
x1 = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
            [5,6,7],[6,7,8],[7,8,9],[8,9,10],
            [9,10,11],[10,11,12],
            [20,30,40],[30,40,50],[40,50,60]])
x2 = array([[10,20,30],[20,30,40],[30,40,50],[40,50,60],
            [50,60,70],[60,70,80],[70,80,90],[80,90,100],
            [90,100,110],[100,110,120],
            [2,3,4],[3,4,5],[4,5,6]])
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x1_pred = array([55,65,75])
x2_pred = array([65,75,85])

# shape 변경
x1 = x1.reshape(13,3)
x2 = x2.reshape(13,3,1)

# 2. 모델
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Input
from tensorflow.keras.layers import concatenate

input1 = Input(shape=(3,))
dense1 = Dense(1024, activation='relu')(input1)
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
dense = Dense(64)(dense)
dense = Dense(32)(dense)
dense = Dense(16)(dense)
output1 = Dense(1)(dense)

model = Model(inputs=[input1, input2], outputs=output1)

model.summary()

# 3. 컴파일, 룬련
model.compile(loss='mse', optimizer='adam')
model.fit([x1, x2], y, epochs=1000, batch_size=4, verbose=2)

# 4. 평가, 예측
loss = model.evaluate([x1,x2], y)
print('loss :', loss)
print('=======================')

x1_pred = array([55,65,75]).reshape(1,3)
x2_pred = array([65,75,85]).reshape(1,3,1)

# 85의 근사치
y_pred = model.predict([x1_pred, x2_pred])
print('y_pred :', y_pred)

# result
# loss : 0.0009221226791851223
# y_pred : [[75.79208]]