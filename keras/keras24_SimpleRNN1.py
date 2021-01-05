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
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN
model = Sequential()
model.add(SimpleRNN(10, activation='relu', input_shape=(3,1))) # (3, 1) -> (timesteps, input_dim)
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))
# LSTM - 3차원, Dense - 2차원

model.summary()

'''
통상적으로 LSTM이 더 좋다
LSTM이 연산을 더 많이 하기 떄문이다
gate는 존재하지 않기 때문에 
param은 120으로 (n + m + 1) * m 으로 계산

activation='tanh'

입력 데이터가 커지면 학습능력이 저하 (앞쪽 데이터의 영향력이 줄어든다)
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