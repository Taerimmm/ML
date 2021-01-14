import numpy as np
import pandas as pd

stock = pd.read_csv('./samsung/삼성전자.csv', index_col=0, header=0, encoding='cp949')

stock.replace(',', '', inplace=True, regex=True)
stock = stock.astype('float32')

stock = stock.iloc[::-1]
print(stock.shape)          # (2400, 14)
print(stock.iloc[-665:-662,:]) # <---- 제거
stock.dropna(inplace=True)
print(stock.shape)          # (2397, 14)
print(stock.iloc[-666:-660,:])

# print(stock.iloc[:-662,:].tail())

stock.iloc[:-662,0:4] = stock.iloc[:-662,0:4]/50.       # 시가, 고가, 저가, 종가 1/50 배 (액면분할)
stock.iloc[:-662,5] = stock.iloc[:-662,5]*50.           # 거래량 50배

print(stock.iloc[-666:-660,:])


print(stock.shape)      # (2397, 14)

print(stock.columns)    # ['시가', '고가', '저가', '종가', '등락률', '거래량', '금액(백만)', '신용비', '개인', '기관','외인(수량)', '외국계', '프로그램', '외인비']

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
x_data = stock.iloc[:,[0,1,2,3,4,8,9]]
y_data = stock['종가']

np.save('./samsung/etc/14day_data.npy', arr=x_data)

print(x_data.columns)         # ['시가', '고가', '저가', '종가', '등락률', '개인', '기관']

print(x_data.shape)         # (2397, 7)
print(y_data.shape)         # (2397,)

def split_x(data,size):
    a = []
    for i in range(data.shape[0] - size + 1):
        a.append(np.array(data.iloc[i:(i+size), 0:len(data.columns)]))
    return  np.array(a)

size = 28
x_data = split_x(x_data,size)

y_target = y_data[size:]

print(x_data[:-1].shape)        # (2369, 28, 7)
print(y_target.shape)           # (2369,)

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

# print(x_train.shape)
# print(x_test.shape)

# print(x_train[-1])
# print(x_test[0])

np.save('./samsung/etc/14day_x_train.npy', arr=x_train)
np.save('./samsung/etc/14day_x_test.npy', arr=x_test)
np.save('./samsung/etc/14day_y_train.npy', arr=y_train)
np.save('./samsung/etc/14day_y_test.npy', arr=y_test)

# 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(x_train.shape[1],x_train.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath = './samsung/etc/14day_model_checkpoint.hdf5'
es = EarlyStopping(monitor='val_loss', patience=200, mode='auto')
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
model.fit(x_train, y_train, epochs=10000, batch_size=4, validation_split=0.2, verbose=2, callbacks=[es,cp])

model.save('./samsung/etc/14day_stock_model.h5')

loss = model.evaluate(x_test, y_test, batch_size=8)
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
# loss : 5067966.0
# RMSE : 2251.2144
# MSE : 5067966.0
# R2 : 0.968343143709883

# 2021-01-14 예측
last = x_data[-1].reshape(1,x_train.shape[1],x_train.shape[2])
a = model.predict(last)
print('\'2021-01-14\'의 종가는', a, '로 예측 됩니다.')

# 예측값
# 83486.67
