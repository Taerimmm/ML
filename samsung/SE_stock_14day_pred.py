import numpy as np
import pandas as pd

stock = np.load('./samsung/etc/14day_data.npy')

# X, Y 나누기
x_data = pd.DataFrame(stock)
y_data = pd.DataFrame(stock).iloc[:,3]

def split_x(data,size):
    a = []
    for i in range(data.shape[0] - size + 1):
        a.append(np.array(data.iloc[i:(i+size), 0:len(data.columns)]))
    return  np.array(a)

size = 28
x_data = split_x(x_data,size)
y_target = y_data[size:]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data[:-1], y_target, test_size=0.2)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) 

x_data_mm = x_data.reshape(x_data.shape[0], x_data.shape[1]*x_data.shape[2])
x_data_mm = scaler.transform(x_data_mm)
x_data = x_data_mm.reshape(x_data.shape[0],x_data.shape[1],x_data.shape[2])

x_train = x_train.reshape(x_train.shape[0], x_data.shape[1], x_data.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_data.shape[1], x_data.shape[2])

from tensorflow.keras.models import load_model

model = load_model('./samsung/etc/14day_stock_model.h5')
# model = load_model('./samsung/etc/14day_model_checkpoint.hdf5')

loss = model.evaluate(x_test, y_test)
print('loss :', loss)

# 2021-01-14 예측
last = x_data[-1].reshape(1,x_train.shape[1],x_train.shape[2])
a = model.predict(last)
print('\'2021-01-14\'의 종가는', a, '로 예측 됩니다.')

# '2021-01-14'의 종가는 [[83608.15]] 로 예측 됩니다.

y_pred = model.predict(x_test)

for i in range(20):
    print('실제 :', y_test.iloc[i+400], '예측 :', y_pred[i+400])
