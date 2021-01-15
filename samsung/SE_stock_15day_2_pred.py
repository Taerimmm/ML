import numpy as np
import pandas as pd

# X, Y 나누기
x_data = pd.DataFrame(np.load('./samsung/etc/15day_data_2.npy'))
y_data = x_data.iloc[:,3]
print(x_data)
print(x_data.shape)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_data)
x_data = scaler.transform(x_data)

def split_x(data,size):
    a = []
    for i in range(data.shape[0] - size + 1):
        a.append(np.array(data.iloc[i:(i+size), 0:len(data.columns)]))
    return  np.array(a)

size = 10
x_data = split_x(pd.DataFrame(x_data),size)

y_target = y_data[size:]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data[:-1], y_target, test_size=0.2)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

x_train = x_train.reshape(x_train.shape[0], x_data.shape[1], x_data.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_data.shape[1], x_data.shape[2])
x_val = x_val.reshape(x_val.shape[0], x_data.shape[1], x_data.shape[2])

from tensorflow.keras.models import load_model

model = load_model('./samsung/etc/15day_model_checkpoint_2.hdf5')
# model = load_model('./samsung/etc/15day_stock_model_2.h5')

loss = model.evaluate(x_test, y_test)
print('loss :', loss)

# 2021-01-15 예측
last = x_data[-1].reshape(1,x_train.shape[1],x_train.shape[2])
a = model.predict(last)
print('\'2021-01-15\'의 종가는', a, '로 예측 됩니다.')

# '2021-01-15'의 종가는 [[88820.67]] 로 예측 됩니다.

y_pred = model.predict(x_test)

for i in range(6):
    print('실제 :', y_test.iloc[i+400], '예측 :', y_pred[i+400])
