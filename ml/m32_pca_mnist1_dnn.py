# m31로 만든 0.95 이상의 n_component=? 를 사용하여 DNN 모델을 만들 것
# mnist dnn 보다 성능 좋게 만들어라
# cnn과 비교

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x = np.append(x_train, x_test, axis=0)
y = np.append(y_train, y_test, axis=0)

print(x.shape, y.shape)     # (70000, 28, 28) (70000,)

x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])

y = to_categorical(y)

pca = PCA()
x = pca.fit_transform(x)

cumsum = np.cumsum(pca.explained_variance_ratio_)
print(cumsum)

d = np.argmax(cumsum >= 0.95) + 1
print('축소된 차원 수 :', d)

pca = PCA(n_components=d)
x = pca.fit_transform(x)
print(x.shape)      # (70000, 154)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/7, random_state=45)

print(x_train.shape, x_test.shape)  # (60000, 154) (10000, 154)

# 2. 모델
model = Sequential()
model.add(Dense(1024, activation='relu', input_shape=(d,)))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', patience=20, mode='auto')
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=7, mode='auto')
model.fit(x_train, y_train, epochs=1000, batch_size=16, validation_split=0.2, verbose=2, callbacks=[es])

loss, acc = model.evaluate(x_test, y_test)
print('loss :', loss)
print('acc :', acc)

# loss : 0.23144932091236115
# acc : 0.9726999998092651

y_pred = model.predict(x_test)

print('==========================')
print('   예상 ' ,'|','   예측  ')
for i in range(10):
    print('  ', np.argmax(y_test[i+40]), '  |', np.argmax(y_pred[i+40]))
