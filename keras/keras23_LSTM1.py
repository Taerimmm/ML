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
model.add(LSTM(10, activation='relu', input_shape=(3,1))) # (3, 1) -> (timesteps, input_dim)
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))
# LSTM - 3차원, Dense - 2차원

model.summary()
'''
LSTM은 4개의 gate(function)으로 구성 - sigmoid + sigmoid + Tanh + sigmoid
param은 480으로 4 * (n + m + 1) * m 으로 계산
n : size of input_dim
m : size of output
gate가 4개이므로 gate의 수만큼 계산이 더 된다.
forget gate, input gate, cell state, output gate

param 계산시 
1. LSTM input_shape
  
   (   ,    ,    ) -> (batch_size, timesteps, input_dim)

activation='tanh', recurrent_activation='sigmoid'
'''


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

# 0.007055305410176516
# [[8.249454]]