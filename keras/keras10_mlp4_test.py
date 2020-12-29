# 실습
# x는 (100, 5) 데이터 임의로 구성
# y는 (100, 2) 데이터 임의로 구성
# 모델을 완성하시오.

# 다 만든 친구들은 predict의 일부 값을 출력하시오.

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


x_pred2 = np.array([101, 401, 101, 500, 351])
print(x_pred2.shape) # (5,)

x_pred2 = np.transpose(x_pred2)
print(x_pred2.shape) # (5, )

x_pred2 = x_pred2.reshape(1, 5)
print(x_pred2.shape) # (1, 5)


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=66)
print(x_train.shape)   # (80, 5)
print(y_train.shape)   # (80, 2)

# 2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# from keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_dim=5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(2))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2)

# 4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test)
print('loss :', loss)
print('mae :', mae)

y_predict = model.predict(x_test)
# print(y_predict[:5])

# [[108.99999 508.00006]
#  [193.99985 593.00006]
#  [104.99999 504.0001 ]
#  [106.00001 505.00003]
#  [152.99998 551.99994]]


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


y_pred2 = model.predict(x_pred2)
print(y_pred2)
# [[200.99991 599.9999 ]]
