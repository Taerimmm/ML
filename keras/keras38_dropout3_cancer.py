# 실습
# Dropout 적용

import numpy as np
from sklearn.datasets import load_breast_cancer

# 1. 데이터
datasets = load_breast_cancer()

print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target
print(x.shape)      # (569, 30)
print(y.shape)      # (569,)

# print(x[:5])
# print(np.max(x), np.min(x))
# print(y)

# 전처리 알어서 해 / train_test_split, minmax
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
model.add(Dense(256, activation='relu', input_shape=(30,)))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
# 마지막 layer의 activation은 sigmoid

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=200, batch_size=1, validation_split=0.2, verbose=2)

loss, acc = model.evaluate(x_test, y_test)
print('loss :', loss)
print('acc :', acc)
print('========================')


y_pred = np.round(model.predict(x_test))

for i in range(10,20):
    true = y_test[i]
    pred = y_pred[i]
    print('실제 :', true, '| 예측 :', pred)
print('========================')


# Dropout 이전
# loss : 0.18867605924606323
# acc : 0.9824561476707458

# Dropout 후
# loss : 1.843904733657837
# acc : 0.9736841917037964