# 과제 및 실습
# Dense
# 전처리 , Early_stopping 등등 다 넣을 것!!
# 데이터 1 ~ 100 
#       x            y
#   1,2,3,4,5        6
#      . . .
# 99,96,97,98,99    100

# Dense / LSTM 비교

# 1. 데이터
import numpy as np

x = np.array(range(1,101))
size = 6

def split_x(a,size):
    return np.array([a[i:(i+size)] for i in [i for i in range(len(x)-size+1)]])

data = split_x(x,size)
print(data)
x = data[:,:5]
y = data[:,5]
print(x.shape)      # (95, 5)
print(y.shape)      # (95,)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=45)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape)
print(x_test.shape) 

# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(5,)))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
# from tensorflow.keras.callbacks import EarlyStopping
# early_stopping = EarlyStopping(monitor='loss', patience=50, mode='auto')
# model.fit(x_train, y_train, epochs=2000, batch_size=4, verbose=2, callbacks=[early_stopping])
model.fit(x_train, y_train, epochs=2000, batch_size=4, verbose=2)

loss = model.evaluate(x_test, y_test)
print('loss :', loss)

# predict를 만들 것
#   96,97,98,99,100    ->  100
#      . . .
# 100,101,102,103,104  ->  105
# 예상 predict 는 (101, 102, 103, 104, 105)
# y_true = np.array([101,102,103,104,105])

x = np.array(range(96,106))
data = split_x(x, size)
x_test = data[:,:5]
x_test = scaler.transform(x_test).reshape(5,5)
y_test = data[:,5]

y_pred = model.predict(x_test)
print('==========================')
print('   예상 ' ,'|','   예측  ')
for i in range(len(y_test)):
    print('  ', y_test[i], '  |', y_pred[i])


# result
# loss : 0.2359008491039276
# ==========================
#    예상  |    예측
#    101   | [100.98943]
#    102   | [101.989456]
#    103   | [102.989525]
#    104   | [103.989624]
#    105   | [104.98978]