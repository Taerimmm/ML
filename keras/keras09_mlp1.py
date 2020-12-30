# 다:1 MLP

import numpy as np

# 1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

print(x.shape)   # (10,) -> 스칼라가 10개

x = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1,2,3,4,5,6,7,8,9,10]])

print(x.shape)   # (2, 10) 


# [[1,2,3],[4,5,6]]               -> (2,3)
# [[1,2],[3,4],[5,6]]             -> (3,2)
# [[[1,2,3],[4,5,6]]]             -> (1,2,3)
# [[1,2,3,4,5,6]]                 -> (1,6)
# [[[1,2],[3,4]],[[5,6],[7,8]]]   -> (2, 2, 2)
# [[1],[2],[3]]                   -> (3, 1)
# [[[1],[2]],[[3],[4]]]           -> (2, 2, 1)

x = np.array([[1,2,3,4,5,6,7,8,9,10],
              [11,12,13,14,15,16,17,18,19,20]]).T
print(x)

x = np.array([[1,2,3,4,5,6,7,8,9,10],
              [11,12,13,14,15,16,17,18,19,20]]).transpose()
print(x)
print(x.shape)   # (10, 2)

# 2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# from keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_dim=2))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x, y, epochs=100, batch_size=1, validation_split=0.2)

# 4. 평가, 예측
loss, mae = model.evaluate(x, y)
print('loss :', loss)
print('mae :', mae)

y_predict = model.predict(x)
# print(y_predict)

# # RMSE 구하기
# from sklearn.metrics import mean_squared_error
# def RMSE(y_test, y_predict):
#     return np.sqrt(mean_squared_error(y_test, y_predict))

# print('RMSE :', RMSE(y_test, y_predict))
# print('mse :', mean_squared_error(y_test, y_predict))

# # R2 구하기
# from sklearn.metrics import r2_score
# r2 = r2_score(y_test, y_predict)
# print('R2 :', r2)
