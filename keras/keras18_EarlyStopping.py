# EarlyStopping
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

'''
* 전처리는 x_train만 한다 *

x_train, x_test 가 disjoint set & total union (?)
훈련은 x_train으로 시킴 -> 손실되는 범위가 있음 -> x_train을 전처리 시켜 0 ~ 1 을 맞춰놓고 -> test에서 범위 0 ~ 1 를 넘어가더라도 파악가능하도록 하는 것이 목적
그렇게 하지 않으면 모든 데이터가 0 ~ 1 사이로 과적합 된다
이 경우 0 ~ 1 의 범위를 벗어나면 값이 틀어진다
따라서 무조건 기준을 x_train 으로 잡고 x_train만 전처리

scaler.fit(x) -> scaler.fit(x_train)
scaler.transform(x) -> scaler.transform(x_train)
scaler.transform(x_test) # test는 train의 기준에 맞춘다

scalaer.transform(x_pred) 도 해줘야함
transform(x_val) ; validation ; validation_data로 나눴을 때 / validation_split할때는 x

x_test와 x_val이 0 ~ 1 범위를 벗어나는 구간이 있기 때문에 범위 0 ~ 1 밖의 부분에서 예측 가능

  전처리  |           X           |                          Y                                            
----------------------------------------------------------------------------------------
  train  |    fit / transform    | Y는 따로 scaling을 하지 않는다 -> 전처리 할 필요가 없다
----------------------------------------------------------------------------------------
   test  |       transform       | Y는 따로 scaling을 하지 않는다 -> 전처리 할 필요가 없다
----------------------------------------------------------------------------------------
   val   |       transform       | Y는 따로 scaling을 하지 않는다 -> 전처리 할 필요가 없다
----------------------------------------------------------------------------------------
 predict |       transform       |                   X -> 데이터가 없다

'''

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=45)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=45)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)


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

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto')

model.fit(x_train, y_train, epochs=2000, batch_size=8, validation_data=(x_val, y_val), verbose=2, callbacks=[early_stopping])

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

# 전처리 후 ; MinMaxScaler - 제대로 전처리
# loss : 8.638400077819824
# MAE : 2.260866165161133
# RMSE : 2.9391154241174844
# R2 : 0.9200725264247241

# 전처리 후 ; MinMaxScaler - validation_data
# loss : 10.164072036743164
# MAE : 2.2508976459503174
# RMSE : 3.1881141213618536
# R2 : 0.9059561241039041

# EARLY_STOPPING
# loss : 9.122064590454102
# MAE : 2.1943304538726807
# RMSE : 3.020275451341502
# R2 : 0.9155973840424944
