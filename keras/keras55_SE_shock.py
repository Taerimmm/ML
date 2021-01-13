import numpy as np
import pandas as pd

stock = pd.read_csv('./Test/samsung_stock.csv', index_col=0, header=0, encoding='cp949')

stock.replace(',' ,'',inplace=True, regex=True)
stock = stock.astype('float32')

stock = stock.iloc[:662, :][::-1]

print(stock.shape)      # (662, 14)
print(stock)

'''
# 상관관계
print(stock.corr())

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.2)
sns.heatmap(data=stock.corr(), square=True, annot=True, cbar=True)
plt.show()
'''

# X, Y 나누기
x_data = stock.drop(['종가'], axis=1)
y_data = stock['종가']

x_data = x_data.iloc[:,[0,1,2,4,5]]

print(x_data.shape)         # (662, 6)
print(y_data.shape)         # (662,)

def split_x(data,size):
    a = []
    for i in range(data.shape[0] - size + 1):
        a.append(np.array(data.iloc[i:(i+size), 0:len(data.columns)]))
    return  np.array(a)

size = 20
x_data = split_x(x_data,size)

y_target = y_data[size:]

print(x_data[:-1].shape)
print(y_target.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data[:-1], y_target, test_size=0.2, shuffle=False)

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

np.save('../data/npy/stock_x_data.npy', arr=x_data)
np.save('../data/npy/stock_x_train.npy', arr=x_train)
np.save('../data/npy/stock_x_test.npy', arr=x_test)
np.save('../data/npy/stock_y_train.npy', arr=y_train)
np.save('../data/npy/stock_y_test.npy', arr=y_test)

# 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# print(type(x_train.shape[1]))
model = Sequential()
model.add(LSTM(256, activation='relu', input_shape=(x_train.shape[1],x_train.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath    = '../data/modelcheckpoint/stock_model.hdf5'
es = EarlyStopping(monitor='val_loss', patience=20, mode='auto')
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
model.fit(x_train, y_train, epochs=10000, batch_size=32, validation_split=0.2, verbose=2, callbacks=[es,cp])

model.save('../data/h5/stock_model.h5')

loss = model.evaluate(x_test, y_test)
print('loss :', loss)

y_pred = model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score
def rmse(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

print('RMSE :', rmse(y_test, y_pred))
print('MSE :', mean_squared_error(y_test, y_pred))

print('R2 :', r2_score(y_test, y_pred))
print('\n', '===========================================','\n')

# Result
# loss: 33503260.0000
# RMSE : 5788.2
# MSE : 33503264.0
# R2 : 0.5757151482925984

# 2021-01-13 예측
last = x_data[-1].reshape(1,x_train.shape[1],x_train.shape[2])
a = model.predict(last)
print('\'2021-01-03\'의 종가는', a, '로 예측 됩니다.')

# 예측값
# 88053.53
