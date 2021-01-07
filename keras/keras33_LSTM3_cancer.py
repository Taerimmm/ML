# sklearn dataset
# LSTM으로 모델링
# Dense와 성능비교
# 이진 분류

import numpy as np

# 1. 데이터
from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

print(x.shape)      # (569, 30)
print(y.shape)      # (569,)

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y.shape)      # (569, 2)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=45)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(30,1)))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(2, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=2)

loss, acc = model.evaluate(x_test, y_test)
print('loss :', loss)
print('acc: ', acc)

# loss : 0.24089103937149048
# acc:  0.9561403393745422

y_pred = model.predict(x_test)

for i in range(10,20):
    true = y_test[i]
    pred = y_pred[i]
    print('실제 :', true, '| 예측 :', pred)
print('========================')

# 실제 : [0. 1.] | 예측 : [2.5850453e-04 9.9972254e-01]
# 실제 : [1. 0.] | 예측 : [0.9877676  0.01149188]
# 실제 : [0. 1.] | 예측 : [9.9207304e-05 9.9989605e-01]
# 실제 : [0. 1.] | 예측 : [0.00593467 0.9936348 ]
# 실제 : [0. 1.] | 예측 : [4.648663e-05 9.999511e-01]
# 실제 : [1. 0.] | 예측 : [9.9998307e-01 1.2000442e-05]
# 실제 : [1. 0.] | 예측 : [2.9480918e-05 9.9996996e-01]
# 실제 : [0. 1.] | 예측 : [3.8736535e-04 9.9959332e-01]
# 실제 : [0. 1.] | 예측 : [0.00113157 0.99878556]
# 실제 : [0. 1.] | 예측 : [8.645841e-05 9.999099e-01]
# ========================