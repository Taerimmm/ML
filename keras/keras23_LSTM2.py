# input_shape / input_length / input_dim

# 1. 데이터
import numpy as np

# 시계열 데이터
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])
y = np.array([4,5,6,7])

print('x.shape: ', x.shape) # (4, 3)
print('y.shape: ', y.shape) # (4,)

x = x.reshape(4, 3, 1)

# 2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
model = Sequential()
# model.add(LSTM(10, activation='relu', input_shape=(3,1))) # (3, 1) -> (timesteps, input_dim)
model.add(LSTM(10, activation='relu', input_length=3, input_dim=1))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))
# LSTM - 3차원, Dense - 2차원

model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)

# 4. 평가, 예측
loss = model.evaluate(x,y)
print(loss)

x_pred = np.array([5,6,7])
x_pred = x_pred.reshape(1,3,1)

result = model.predict(x_pred)
print(result)
