import numpy as np

# 1. 데이터
x = np.array([range(1,101), range(301,401), range(1,101), range(400,500), range(251,351)])
y = np.array([range(101, 201), range(500,600)])
print(x.shape)   # (5, 100) 
print(y.shape)   # (2, 100)

x = np.transpose(x)
y = np.transpose(y)
print(x.shape)   # (100, 5)
print(y.shape)   # (100, 2)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=66)
print(x_train.shape)   # (80, 5)
print(y_train.shape)   # (80, 2)

# 2. 모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
# from keras.layers import Dense

input1 = Input(shape=(5,)) # input layer
dense1 = Dense(5, activation='relu')(input1)
dense2 = Dense(3)(dense1)
dense3 = Dense(4)(dense2)
outputs = Dense(2)(dense3)
model = Model(inputs = input1, outputs = outputs)
model.summary()

# model = Sequential()
# # model.add(Dense(10, input_dim=5))
# model.add(Dense(5, input_shape=(1,), activation='relu'))
# # 가장 앞의 데이터의 개수 무시
# model.add(Dense(3))
# model.add(Dense(4))
# model.add(Dense(1))
# model.summary()


# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, verbose=1)

# 4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test)
print('loss :', loss)
print('mae :', mae)

y_predict = model.predict(x_test)
# print(y_predict)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print('RMSE :', RMSE(y_test, y_predict))
print('mse :', mean_squared_error(y_test, y_predict))

# R2 구하기
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('R2 :', r2)
