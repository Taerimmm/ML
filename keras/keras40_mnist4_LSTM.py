# 주말 과제
# Dence 모델로 구성 input_shape=(28*28,1)
# Dence 모델로 구성 input_shape=(28*14,2)
# Dence 모델로 구성 input_shape=(28*7,4)

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)     # (60000, 28, 28), (60000,) <- 흑백
print(x_test.shape, y_test.shape)       # (10000, 28, 28), (10000,)

print(x_train[0])
print(y_train[0])

print(x_train[0].shape)         # (28, 28)

x_train = x_train.reshape(60000, 28 * 4, 7).astype('float32')/255.
x_test = x_test.reshape(10000, 28 * 4, 7)/255. 

print(x_train.shape, x_test.shape)     # (60000, 112, 7) (10000, 112, 7)

# OneHotEncoding
# 여러분이 하시오!!!
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train)
print(y_test)

# 실습!! 완성하시오!!! 
# 지표는 acc /// 0.985 이상

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

model = Sequential()
model.add(LSTM(128, activation='relu', input_shape=(112,7)))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='auto')
model.fit(x_train, y_train, epochs=4000, batch_size=16, validation_split=0.2, verbose=2, callbacks=[early_stopping])

loss, acc = model.evaluate(x_test, y_test)
print('loss :', loss)
print('acc :', acc)

# loss : 0.09020749479532242
# acc : 0.9750999808311462

# 응용
# y_test 10개와 10개를 출력하시오


y_pred = model.predict(x_test)

print(y_pred)
print('==========================')
print('   예상 ' ,'|','   예측  ')
for i in range(10):
    print('  ', np.argmax(y_test[i+40]), '  |', np.argmax(y_pred[i+40]))
