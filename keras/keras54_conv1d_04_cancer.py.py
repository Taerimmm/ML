# 실습
# Conv1d로 코딩
import numpy as np

# 1. 데이터
from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

print(x.shape)      # (569, 30)
print(y.shape)      # (569,)

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y.shape)      # (569, 2)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=45)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
model = Sequential()
model.add(Conv1D(filters=256, kernel_size=2, padding='same', strides=1, input_shape=(30,1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv1D(128, 2, padding='same', strides=1))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=20, mode='auto')
model.fit(x_train, y_train, epochs=1000, batch_size=8, validation_split=0.20, verbose=2, callbacks=[es])

loss, acc = model.evaluate(x_test, y_test)
print('loss :', loss)
print('acc: ', acc)

# result
# loss : 0.38370904326438904
# acc:  0.9824561476707458

y_pred = model.predict(x_test)

for i in range(10,20):
    true = np.argmax(y_test[i])
    pred = np.argmax(y_pred[i])
    print('실제 :', true, '| 예측 :', pred)
