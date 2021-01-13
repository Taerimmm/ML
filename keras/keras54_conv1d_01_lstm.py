# 실습
# Conv1d로 코딩
import numpy as np

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
              [5,6,7], [6,7,8], [7,8,9], [8,9,10],
              [9,10,11], [10,11,12],
              [20,30,40], [30,40,50], [40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70]) 

x_pred = np.array([50,60,70])

print(x.shape)      # (13, 3)
print(y.shape)      # (13,)

x = x.reshape(13,3,1)
print(x.shape)

print(x_pred.shape)
x_pred = x_pred.reshape(1,3,1)
print(x_pred.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, padding='same', strides=1, input_shape=(3,1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=500, batch_size=1, validation_split=0.2, verbose=2)

loss = model.evaluate(x, y)
print('loss :', loss)

# Result
# loss : 69.89508819580078

y_pred = model.predict(x_pred)
print('y_pred :', y_pred)

# y_pred : [[61.474247]]