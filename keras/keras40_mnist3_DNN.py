# 주말 과제
# Dence 모델로 구성 input_shape=(28*28, )

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)     # (60000, 28, 28), (60000,) <- 흑백
print(x_test.shape, y_test.shape)       # (10000, 28, 28), (10000,)

print(x_train[0])
print(y_train[0])

print(x_train[0].shape)         # (28, 28)

x_train = x_train.reshape(60000,28 *28).astype('float32')/255.
x_test = x_test.reshape(10000, 28 * 28)/255. 

print(x_train.shape, x_test.shape)     # (60000, 784), (10000, 784)

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
from tensorflow.keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(1024, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')
model.fit(x_train, y_train, epochs=4000, batch_size=32, validation_split=0.2, verbose=2, callbacks=[early_stopping])

loss, acc = model.evaluate(x_test, y_test)
print('loss :', loss)
print('acc :', acc)

# loss : 0.15171925723552704
# acc : 0.984499990940094

# 응용
# y_test 10개와 10개를 출력하시오


y_pred = model.predict(x_test)

print(y_pred)
print('==========================')
print('   예상 ' ,'|','   예측  ')
for i in range(10):
    print('  ', np.argmax(y_test[i+40]), '  |', np.argmax(y_pred[i+40]))
