from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1. 데이터
x = np.array(range(1,101))
y = np.array(range(101,201))

# x_train = x[:60]
# x_val = x[60:80]
# x_test = x[80:]
# # list slicing

# y_train = y[:60]
# y_val = y[60:80]
# y_test = y[80:]
# # list slicing

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True)
# split할때 train set과 test set의 Weight나 loss의 변동을 줄이기 위해 랜덤성을 준다 - 범위를 유사하게 하기 위해
print(x_train)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# 2. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(3))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(300))
model.add(Dense(4))
model.add(Dense(300))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=1000)

# 4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test)
print('loss :', loss)
print('mae :', mae)

y_predict = model.predict(x_test)
print(y_predict)

# shuffle = False
# loss : 1.9208528101444244e-09
# mae : 2.975463939947076e-05

# shuffle = True
# loss : 2.1536834815538697e-10
# mae : 1.068115216185106e-05

# i = 0
# while 1:
#   print(y_test[i], y_predict[i])
#   i += 1
#   if i == len(y_test):
#       break