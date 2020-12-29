import numpy as np

# 1. 데이터
x = np.array([range(100), range(101,201), range(401,501), range(901,1001)]).T
y = np.array([range(50,150), range(5,505,5)]).T

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=45)

# 2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

# functional
inputs = Input(shape=(4,))
dense1 = Dense(10, activation='relu')(inputs)
dense2 = Dense(20)(dense1)
dense3 = Dense(10)(dense2)
dense4 = Dense(50)(dense3)
outputs = Dense(2)(dense4)
model = Model(inputs=inputs, outputs=outputs)

# sequential
# model = Sequential()
# model.add(Dense(10, input_shape=(4,), activation='relu'))
# model.add(Dense(20))
# model.add(Dense(10))
# model.add(Dense(50))
# model.add(Dense(20))
# model.add(Dense(2))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, verbose=2)

# 4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test)
print('loss :', loss)
print('MAE :', mae)

y_predict = model.predict(x_test)
# print(y_predict[:5])

# RMSE
from sklearn.metrics import mean_squared_error
def  rmse(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print('RMSE :', rmse(y_test, y_predict))
print('MSE :', mean_squared_error(y_test, y_predict))

# R2
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('R2 :', r2)