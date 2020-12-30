import numpy as np
import tensorflow as tf

# 1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_dim = 1, activation ='linear'))
model.add(Dense(29, activation ='linear'))
model.add(Dense(20))
model.add(Dense(15))
model.add(Dense(9, name='denss'))
model.add(Dense(20))
model.add(Dense(1))

model.summary()
# shape - 노드의 개수
# param - 연산된 수
# bias 가 더해지기 때문에 노드의 개수가 +1이 되고 다음 레이어의 노드 수와 곱해져 param 수가 된다.

# 실습 2 + 과제
# ensemble 1, 2, 3, 4 에 대해 서머리를 계산하고
# 이해한 것을 과제로 제출할 것
# layer를 만들 때 'name' 이란 놈에 대해 확인하고 설명할 것 
# name을 반드시 써야 할 때가 있다. 그때를 말하라.