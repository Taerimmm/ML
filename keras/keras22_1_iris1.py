import numpy as np
from sklearn.datasets import load_iris

# 1. 데이터
# x, y = load_iris(return_X_y=True)

dataset = load_iris()
x = dataset.data
y = dataset.target
# print(dataset.DESCR)
# print(dataset.feature_names)

print(x.shape)  # (150, 4)
print(y.shape)  # (150,)
# print(x[:5])
# print(y)
# print(np.max(x), np.min(x))

# 원핫인코딩 y에 대한 전처리
'''
to_categoricaal 은 무조건 0부터 시작
만약 0 값이 없다면 0을 넣어줘야한다
'''
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

# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(4,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))
# 다중 분류에서는 분류하고자 하는 숫자의 개수만큼 노드를 잡는다
# activation = 'softmax'

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, verbose=2)

# 4. 평가
loss, acc = model.evaluate(x_test, y_test)
print('loss :', loss)
print('acc :', acc)

# loss : 0.08805288374423981
# acc : 0.9666666388511658

y_pred = model.predict(x_test[-5:-1])
# print(y_pred)
print(y_test[-5:-1])

# 결과치 나오게 코딩할것 # argmax
print(np.argmax(y_pred, axis=1))