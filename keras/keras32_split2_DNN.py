import numpy as np

a = np.array(range(1,11))
size = 5

# 모델을 구성하시오
def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

dataset = split_x(a,size)
x = dataset[:,:4] # (6, 4)
y = dataset[:,4]  # (6,)

print(dataset)
print(x)
print(y)
print(x.shape)
print(y.shape)
x = x.reshape(6,4)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
input1 = Input(shape=(4,))
dense = Dense(64, activation='relu')(input1)
dense = Dense(32)(dense)
dense = Dense(16)(dense)
output1 = Dense(1)(dense)
model = Model(inputs=input1, outputs=output1)

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=200, batch_size=4, verbose=2)

loss = model.evaluate(x,y)
print('loss :', loss)

x_pred = np.array([2,4,6,8]).reshape(1,4)
y_pred = model.predict(x_pred)
print('y_pred :', y_pred)

# result
# loss : 0.004633862990885973
# y_pred : [[8.814702]]

x_pred = dataset[-1,1:].reshape(1,4)
print(x_pred)
y_pred = model.predict(x_pred)
print('y_pred :', y_pred)

# y_pred : [[11.126733]]