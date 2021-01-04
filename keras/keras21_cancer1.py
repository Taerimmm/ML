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
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(30,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# 마지막 layer의 activation은 sigmoid

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# 이진 분류인 경우 binary_crossentropy를 loss로 사용
# metrics에 가급적 acc 사용
model.fit(x_train, y_train, epochs=200, batch_size=1, validation_split=0.2, verbose=2)

loss, acc = model.evaluate(x_test, y_test)
print('loss :', loss)
print('acc :', acc)
print('========================')

# 실습 1. acc 0.985 이상 올릴 것

# loss : 0.6525935530662537
# acc : 0.9912280440330505

# 실습 2. predict 출력해 볼 것

y_pred = np.round(model.predict(x_test))

for i in range(10,20):
    true = y_test[i]
    pred = y_pred[i]
    print('실제 :', true, '| 예측 :', pred)
print('========================')
