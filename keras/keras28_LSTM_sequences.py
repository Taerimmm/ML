# keras23_3 을 copy 해서
# LSTM층 두개를 만들기 !!!

# input_shape / input_length / input_dim

# 1. 데이터
import numpy as np

# 시계열 데이터
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])
y = np.array([4,5,6,7])

print('x.shape: ', x.shape) # (4, 3)
print('y.shape: ', y.shape) # (4,)

# x = x.reshape(13, 3, 1)
x = x.reshape(x.shape[0], x.shape[1], 1)
print(x)

# 2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(3,1), return_sequences=True))
model.add(LSTM(10, activation='relu')) # LSTM 1층에서 출력한 값이 시계열 값 x 
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))
# LSTM - 3차원, Dense - 2차원

model.summary()

'''
LSTM layer 2 의 param이 840인 이유
-> LSTM layer 1 에서 return_sequences=True로 3차원으로 출력된다
   10개의 input이 전달되므로 layer 2 에서 size_input 이 10으로 된다.
Ex. input_shape=(100,5) --> output_shape = (None, 100, 10) (return_sequences=True)

lstm은 input을 3차원으로 받아들이지만 output은 2차원 (return_sequences=False의 경우)

- 4 * (input_size + output_size + 1(bias)) * output_size
- 4 * (10 + 10 + 1) * 10 = 840
'''


# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)

# 4. 평가, 예측
loss = model.evaluate(x,y)
print(loss)

x_pred = np.array([5,6,7])
x_pred = x_pred.reshape(1,3,1)
print(x_pred)

result = model.predict(x_pred)
print(result)

# 1 layer LSTM
# 0.012067914009094238
# [[8.105696]]

# 2 layer LSTM
# 0.002773623913526535
# [[8.133904]]


'''
LSTM 2개 한것이 더 안좋다
Why? LSTM 1층에서 출력한 값이 시계열 값이 아니기 떄문
데이터에 따라서 약간 틀릴 수 있다.
통상적으로 2개이상일 경우 성능이 떨어진다.
'''