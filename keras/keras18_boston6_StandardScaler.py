import numpy as np

# skleran / tensorflow / keras 에서 교육용 데이터 제공
from sklearn.datasets import load_boston

# 1. 데이터
dataset = load_boston()
x = dataset.data
y = dataset.target
print(x.shape)   # (506, 13)
print(y.shape)   # (506, )

print('************************************')
print(x[:5])
print(y[:10])
print('************************************')

print(np.max(x), np.min(x))   # 711.0 / 0.0
print(dataset.feature_names)
# print(dataset.DESCR)

# 데이터 전처리 (MinMaxScaler ; (x - min) / (max - min) -> 0 <= x' <= 1)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

# Minmax
# print(np.max(x), np.min(x))   # 711.0 / 0.0  ->  1.0 / 0.0
# print(np.max(x[0]))           # 0.99999999999999999

#
print(np.max(x), np.min(x))   # 9.933930601860268 -3.9071933049810337
print(np.max(x[0]))           # 0.44105193260704206


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=45)

# 2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape=(13,))
dense1 = Dense(128, activation='relu')(input1)
dense2 = Dense(64, activation='relu')(dense1)
dense3 = Dense(64, activation='relu')(dense2)
dense4 = Dense(64, activation='relu')(dense3)
dense5 = Dense(64, activation='relu')(dense4)
output1 = Dense(1)(dense5)

model = Model(inputs=input1, outputs=output1)

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=200, batch_size=8, validation_split=0.2, verbose=2)

# 4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test)
print('loss :', loss)
print('MAE :', mae)

y_predict = model.predict(x_test)

# RMSE, r2
from sklearn.metrics import mean_squared_error, r2_score
def rmse(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE :', rmse(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print('R2 :', r2)

# 전처리 전
# loss : 22.40496253967285
# MAE : 3.470602035522461
# RMSE : 4.733387950906617
# R2 : 0.7926963238194815

# 전처리 후
# loss : 21.25548553466797
# MAE : 3.376149892807007
# RMSE : 4.610367151541528
# R2 : 0.8033319433683961

# 전처리 후 ; MinMaxScaler - X 통채로 변환
# loss : 8.773988723754883
# MAE : 2.226693630218506
# RMSE : 2.962091944054004
# R2 : 0.9188179765594677

# 전처리 후 ; StandardScaler - X 통채로 변환
# loss : 8.085567474365234
# MAE : 2.14711332321167
# RMSE : 2.8435130294316653
# R2 : 0.9251876586121267