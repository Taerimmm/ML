import numpy as np
import pandas as pd

# 삼성전자
stock = pd.read_csv('./samsung/삼성전자.csv', index_col=0, header=0, encoding='cp949')
stock1 = pd.read_csv('./samsung/삼성전자2.csv', index_col=0, header=0, encoding='cp949')
stock2 = pd.read_csv('./samsung/삼성전자3.csv', index_col=0, header=0, encoding='cp949')

stock.drop("2021-01-13", axis=0, inplace=True)
stock1 = stock1.iloc[:2,:]
stock = pd.concat([stock1, stock], join='inner')

stock2.rename(index=dict(zip([x for x in stock2.index], [x.replace('/','-') for x in stock2.index])), inplace=True)
stock.drop("2021-01-14", axis=0, inplace=True)
stock2 = stock2.iloc[:2,:]
stock = pd.concat([stock2, stock], join='inner')

print(stock)
print(stock.shape)      # (2402, 14)

stock.replace(',', '', inplace=True, regex=True)
stock = stock.astype('float32')
stock = stock.iloc[::-1] 

print(stock.iloc[-667:-664,:]) # <---- 제거
stock.dropna(inplace=True)
print(stock.shape)          # (2399, 14)
print(stock.iloc[-666:-661,:])

print(stock.iloc[:-664,:])

stock.iloc[:-664,0:4] = stock.iloc[:-663,0:4]/50.       # 시가, 고가, 저가, 종가 1/50 배 (액면분할)
stock.iloc[:-664,5] = stock.iloc[:-663,5]*50.           # 거래량 50배

print(stock.iloc[-666:-660,:])

# 코스닥 인버스
kos = pd.read_csv('./samsung/코스닥_인버스.csv', index_col=0, header=0, encoding='cp949')

kos.rename(index=dict(zip([x for x in kos.index], [x.replace('/','-') for x in kos.index])), inplace=True)
kos.drop('전일비', axis=1, inplace=True)
kos.rename(columns={'Unnamed: 6' : '전일비'}, inplace=True)

kos.replace(',', '', inplace=True, regex=True)
kos = kos.astype('float32')
kos = kos.iloc[::-1] 

print(kos.shape)      # (1088, 15)
print(kos.iloc[-667:-664,:]) # <---- 제거
kos.drop(['2018-04-30','2018-05-02','2018-05-03'], axis=0, inplace=True)
print(kos.shape)          # (1085, 14)
print(kos.iloc[-666:-661,:])

# X, Y 나누기
print(stock.columns)    # ['시가', '고가', '저가', '종가', '등락률', '거래량', '금액(백만)', '신용비', '개인', '기관','외인(수량)', '외국계', '프로그램', '외인비']
print(kos.columns)      # ['시가', '고가', '저가', '종가', '전일비', '등락률', '거래량', '금액(백만)', '신용비', '개인','기관', '외인(수량)', '외국계', '프로그램', '외인비']

x1_data = stock.iloc[:,[0,1,2,3,4,8,9]]
x2_data = kos.iloc[:,[0,1,2,3,4,5,9,10]]
y_data = stock['시가']

print(x1_data.columns)    # ['시가', '고가', '저가', '종가', '등락률', '개인', '기관']
print(x2_data.columns)      # ['시가', '고가', '저가', '종가', '전일비', '등락률', '개인','기관']

np.save('./samsung/etc/19day_x1_data.npy', arr=x1_data)
np.save('./samsung/etc/19day_x2_data.npy', arr=x2_data)

print(x1_data.shape)         # (2399, 7)
print(x2_data.shape)         # (1085, 8)
print(y_data.shape)          # (2399,)

x1_data = x1_data[-1085:]
y_data = y_data[-1085:]

print(x1_data.shape)         # (1085, 7)
print(x2_data.shape)         # (1085, 8)
print(y_data.shape)          # (1085,)

print(x1_data.head(2))
print(x2_data.head(2))

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

print(x1_data.shape)    # (1066, 20, 7)
print(x2_data.shape)    # (1066, 20, 8)
print(y_target.shape)   # (1065, 2, 1)

from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1_data[:-1], x2_data[:-1], y_target, test_size=0.2, random_state=45)
x1_train, x1_val, x2_train, x2_val, y_train, y_val = train_test_split(x1_train, x2_train, y_train, test_size=0.2, random_state=45)

print(x1_train.shape, x1_test.shape, x1_val.shape)
print(x2_train.shape, x2_test.shape, x2_val.shape)
print(y_train.shape, y_test.shape, y_val.shape)

# 모델
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, GRU, Conv1D, MaxPooling1D, Flatten, Dropout, Concatenate

input1 = Input(shape=(x1_data.shape[1], x1_data.shape[2]))
layer1 = GRU(64)(input1)
layer1 = Dropout(0.2)(layer1)
layer1 = Dense(32)(layer1)

input2 = Input(shape=(x2_data.shape[1], x2_data.shape[2]))
layer2 = Conv1D(16, 3, padding='same', strides=1, activation='relu')(input2)
layer2 = MaxPooling1D(2)(layer2)
layer2 = Dropout(0.2)(layer2)
layer2 = Flatten()(layer2)
layer2 = Dense(16)(layer2)

merge = Concatenate()([layer1, layer2])
merge = Dense(64, activation='relu')(merge)
merge = Dense(64, activation='relu')(merge)
output1 = Dense(2, activation='relu')(merge)

model = Model(inputs=[input1, input2], outputs=output1)

model.summary()

# 훈련
model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
modelpath = './samsung/etc/19day_model_checkpoint.hdf5'
es = EarlyStopping(monitor='val_loss', patience=200, mode='auto')
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.5, verbose=1)
model.fit([x1_train,x2_train], y_train, epochs=10000, batch_size=4, validation_data=([x1_val,x2_val], y_val), verbose=2, callbacks=[es,cp,reduce_lr])

model.save('./samsung/etc/19day_stock_model.h5')

# 예측
loss = model.evaluate([x1_test,x2_test], y_test, batch_size=8)
print('loss :', loss)

x1_last = x1_data[-1].reshape(1,x1_data.shape[1],x1_data.shape[2])
x2_last = x2_data[-1].reshape(1,x2_data.shape[1],x2_data.shape[2])
a = model.predict([x1_last,x2_last])
print('\"2021-01-18\"의 시가는', a[0][0], '로 예측 됩니다.')
print('\"2021-01-19\"의 시가는', a[0][1], '로 예측 됩니다.')

# "2021-01-18"의 시가는 87079.23 로 예측 됩니다.
# "2021-01-19"의 시가는 87598.63 로 예측 됩니다.
