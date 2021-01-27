import numpy as np
from sklearn.datasets import load_iris

# 1. 데이터
dataset = load_iris()
x = dataset.data
y = dataset.target

print(x.shape)  # (150, 4)
print(y.shape)  # (150,)
'''
# 원핫인코딩 y에 대한 전처리 (deep learning case)
from tensorflow.keras.utils import to_categorical 
# from keras.utils.np_utils import to_categorical
y = to_categorical(y)
print(y)
print(y.shape)  # (150, 3)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
'''

'''
머신러닝에서 OneHotEncoding 해줄 필요 없다
'''

# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.svm import LinearSVC

# model = Sequential()
# model.add(Dense(10, activation='relu', input_shape=(4,)))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(3, activation='softmax'))

model = LinearSVC()

# 3. 컴파일, 훈련
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, verbose=2)

model.fit(x,y)

# 4. 평가, 예측
# loss, acc = model.evaluate(x_test, y_test)
# print('loss :', loss)
# print('acc :', acc)

# loss : 0.08805288374423981
# acc : 0.9666666388511658

result = model.score(x,y)
print('score(acc) :',result)
# score(acc) : 0.9666666666666667

y_pred = model.predict(x[-5:-1])
# print(y_pred)
print(y[-5:-1])
print(y_pred)
