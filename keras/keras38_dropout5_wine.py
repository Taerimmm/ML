# 실습
# Dropout 적용

import numpy as np
from sklearn.datasets import load_wine

dataset = load_wine()
x = dataset.data
y = dataset.target

print(x.shape)    # (178, 13)
print(y.shape)    # (178,)

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y.shape)    # (178, 3)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=45)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(13,)))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=200, batch_size=4, verbose=2)

loss, acc = model.evaluate(x_test, y_test)
print('loss :', loss)
print('acc :', acc)

y_pred = model.predict(x_test[-5:-1])
# print(y_pred)
print(y_test[-5:-1])

# 결과치 나오게 코딩할것 # argmax
print(np.argmax(y_pred, axis=1))

# Dropout 이전
# loss : 0.13868051767349243
# acc : 0.9722222089767456

# Dropout 후
# loss : 0.07341800630092621
# acc : 0.9722222089767456