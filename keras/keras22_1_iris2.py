import numpy as np

from sklearn.datasets import load_iris
dataset = load_iris()
x = dataset.data
y = dataset.target
print(y)
print(y.shape)
# sklearn one-hot encoding
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
labels = y
label_encoder = LabelEncoder()
label_ids = label_encoder.fit_transform(labels)

encoder = OneHotEncoder(sparse=False)
reshaped = label_ids.reshape(len(label_ids), 1)
y = encoder.fit_transform(reshaped)
print(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(4,)))
model.add(Dense(64))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, verbose=2)

loss, acc = model.evaluate(x_test, y_test)
print('loss :', loss)
print('acc :', acc)

# loss : 0.17945368587970734
# acc : 0.9333333373069763

y_pred = model.predict(x_test[-5:-1])
# print(y_pred)
print(y_test[-5:-1])

# 결과치 나오게 코딩할것 # argmax
print(np.argmax(y_pred, axis=1))