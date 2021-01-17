import numpy as np
import pandas as pd

# X, Y 나누기
'''
x1_data = pd.DataFrame(np.load('./samsung/etc/19day_data_1.npy'))
x2_data = pd.DataFrame(np.load('./samsung/etc/19day_data_1.npy))
'''
x1_data = pd.DataFrame(np.load('/content/drive/My Drive/samsung/etc/19day_data_1.npy'))
x2_data = pd.DataFrame(np.load('/content/drive/My Drive/samsung/etc/19day_data_2.npy'))
y_data = x1_data.iloc[:,0]

x1_data = x1_data[-1085:]
y_data = y_data[-1085:]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x1_data = scaler.fit_transform(x1_data)
x2_data = scaler.fit_transform(x2_data)

def split_x(data,size):
    a = []
    for i in range(data.shape[0] - size + 1):
        a.append(np.array(data.iloc[i:(i+size), 0:len(data.columns)]))
    return  np.array(a)

size_x = 20
x1_data = split_x(pd.DataFrame(x1_data),size_x)
x2_data = split_x(pd.DataFrame(x2_data),size_x)

size_y = 2
y_data = split_x(pd.DataFrame(y_data),size_y)
y_target = y_data[size_x-size_y+1:] 

from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1_data[:-1], x2_data[:-1], y_target, test_size=0.2, random_state=45)
x1_train, x1_val, x2_train, x2_val, y_train, y_val = train_test_split(x1_train, x2_train, y_train, test_size=0.2, random_state=45)

from tensorflow.keras.models import load_model
'''
# model = load_model('./samsung/etc/15day_model_checkpoint_2.hdf5')
model = load_model('./samsung/etc/15day_stock_model_2.h5')
'''
# EarlyStopping case
model = load_model('/content/drive/My Drive/samsung/etc/19day_stock_model.h5')

loss = model.evaluate([x1_test,x2_test], y_test)
print('loss :', loss)

# 2021-01-19 예측
print('model.h5의 결과')
x1_last = x1_data[-1].reshape(1,x1_data.shape[1],x1_data.shape[2])
x2_last = x2_data[-1].reshape(1,x2_data.shape[1],x2_data.shape[2])
a = model.predict([x1_last,x2_last])
print('\"2021-01-18\"의 시가는', a[0][0], '로 예측 됩니다.')
print('\"2021-01-19\"의 시가는', a[0][1], '로 예측 됩니다.')

y_pred = model.predict([x1_test,x2_test])

print(y_pred.shape, y_test.shape)
print('==================================')
print('시가 예측')

for i in range(6):
    print('실제 :', y_test[i+200][0][0], '예측 :', y_pred[i+200][0])


# ModelCheckpoint case
model = load_model('/content/drive/My Drive/samsung/etc/19day_model_checkpoint.hdf5')
loss = model.evaluate([x1_test,x2_test], y_test)
print('loss :', loss)

# 2021-01-19 예측
print('model_checkpoint.hdf5의 결과')
x1_last = x1_data[-1].reshape(1,x1_data.shape[1],x1_data.shape[2])
x2_last = x2_data[-1].reshape(1,x2_data.shape[1],x2_data.shape[2])
a = model.predict([x1_last,x2_last])
print('\"2021-01-18\"의 시가는', a[0][0], '로 예측 됩니다.')
print('\"2021-01-19\"의 시가는', a[0][1], '로 예측 됩니다.')

y_pred = model.predict([x1_test,x2_test])

print(y_pred.shape, y_test.shape)
print('==================================')
print('시가 예측')

for i in range(6):
    print('실제 :', y_test[i+200][0][0], '예측 :', y_pred[i+200][0])
