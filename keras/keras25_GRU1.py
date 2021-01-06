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
from tensorflow.keras.layers import Dense, LSTM, GRU
model = Sequential()
model.add(GRU(10, activation='relu', input_shape=(3,1))) # (3, 1) -> (timesteps, input_dim)
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))
# LSTM - 3차원, Dense - 2차원

model.summary()
'''
GPU는 기존의 LSTM에서 cell state를 줄인 형태
cell state의 역할을 다음 출력 h에서 역할을 함께 한다
param은 390으로  3 * (n + m + 1 + 1) - sigmoid 2 + tanh 1
n : size of input_dim
m : size of output
1 : bias
1 : cell_state

성능은 LSTM과 유사
속도는 더 빠르다
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

# LSTM
# 0.007055305410176516
# [[8.249454]]

# SiimpleRNN
# 4.5204087655292824e-05
# [[8.00936]]

# GRU
# 0.0014009997248649597
# [[7.8484435]]