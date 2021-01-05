import numpy as np

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
              [5,6,7], [6,7,8], [7,8,9], [8,9,10],
              [9,10,11], [10,11,12],
              [20,30,40], [30,40,50], [40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70]) 

x_pred = np.array([50,60,70])

# 코딩하시오 !!! LSTM
# 나는 80을 원하고 있다.

print(x.shape)      # (13, 3)
print(y.shape)      # (13,)

x = x.reshape(13,3,1)
print(x.shape)

print(x_pred.shape)
x_pred = x_pred.reshape(1,3,1)
print(x_pred.shape)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input

input1 = Input(shape=(3,1))
dense = LSTM(1024, activation='relu')(input1)
dense = Dense(512, activation='relu')(dense)
dense = Dense(256, activation='relu')(dense)
dense = Dense(128, activation='relu')(dense)
dense = Dense(64, activation='relu')(dense)
dense = Dense(64, activation='relu')(dense)
dense = Dense(64, activation='relu')(dense)
dense = Dense(64, activation='relu')(dense)
dense = Dense(64, activation='relu')(dense)
output1 = Dense(1)(dense)
model = Model(inputs=input1, outputs=output1)

# model = Sequential()
# model.add(LSTM(1024, activation='relu', input_shape=(3,1)))
# model.add(Dense(512, activation='relu'))
# model.add(Dense(256, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=500, batch_size=1, verbose=2)

loss = model.evaluate(x, y)
print('loss :', loss)

y_pred = model.predict(x_pred)
print('y_pred :', y_pred)

# Sequential
# loss : 0.0026478709187358618
# y_pred : [[80.69948]]

# Functional
# loss : 0.0014337139436975121
# y_pred : [[80.34424]]