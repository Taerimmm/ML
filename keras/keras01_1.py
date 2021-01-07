import numpy as np
import tensorflow as tf

# 1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(5, input_dim = 1, activation ='linear'))
model.add(Dense(3, activation ='linear'))
model.add(Dense(4))
model.add(Dense(1))

# 3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam, SGD
model.compile(loss='mse', optimizer=Adam(learning_rate=0.1))
# model.compile(loss='mse', optimizer=SGD(learning_rate=0.1))

model.fit(x, y, epochs = 100, batch_size = 1)

# 4. 평가, 예측
loss = model.evaluate(x, y, batch_size = 1)
print('loss :', loss)

x_pred = np.array([4])
# result = model.predict([4])
result = model.predict(x_pred)
print('result :', result)