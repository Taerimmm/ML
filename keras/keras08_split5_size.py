from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1. 데이터
x = np.array(range(1,101))
y = np.array(range(101,201))

from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, test_size=0.2, shuffle=False)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.5, test_size=0.2, shuffle=False)

# train_test_split에서 train_size와 test_size는 0과 1 사이의 수여야 하고 합도 0과 1 사이의 수여야 한다
# 하지만 1보다 작은 경우는 가능하다
# 이 경우 shuffle = False의 경우를 보면 알 수 있듯이 앞에서부터 size만큼 split을 하게 된다

print(x_train)
print(x_test)

'''
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

print(x_train.shape)
print(x_val.shape)
print(x_test.shape)
print(y_train.shape)
print(y_val.shape)
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
model.fit(x_train, y_train, epochs=1000, validation_data=(x_val, y_val))
# validation 16개

# 4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test)
print('loss :', loss)
print('mae :', mae)

y_predict = model.predict(x_test)

# shuffle = False
# loss : 1.9208528101444244e-09
# mae : 2.975463939947076e-05

# shuffle = True
# loss : 2.1536834815538697e-10
# mae : 1.068115216185106e-05

# validation_data = x_val, y_val
# loss : 1.1408701761084217e-09
# mae : 2.136230432370212e-05
# 데이터 크기가 작아 훈련량 자체가 떨어져 loss가 떨어진다

# i = 0
# while 1:
#     print(y_test[i], y_predict[i])
#     i += 1
#     if i == len(y_test):
#         break

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print('RMSE :', RMSE(y_test, y_predict))
print('mse :', mean_squared_error(y_test, y_predict))


from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('R2 :', r2)

# loss : 5.326000751537663e-10
# mae : 1.411438006471144e-05
# RMSE : 2.3078130282980613e-05
# mse : 5.326000973582268e-10
# R2 : 0.9999999999993362
'''