# EarlyStopping
import numpy as np

# skleran / tensorflow / keras 에서 교육용 데이터 제공
from sklearn.datasets import load_boston

# 1. 데이터
dataset = load_boston()
x = dataset.data
y = dataset.target
print(x.shape)   # (506, 13)
print(y.shape)   # (506, )

print('************************************')
print(x[:5])
print(y[:10])
print('************************************')

print(np.max(x), np.min(x))   # 711.0 / 0.0
print(dataset.feature_names)
# print(dataset.DESCR)

# 데이터 전처리 (MinMaxScaler ; (x - min) / (max - min) -> 0 <= x' <= 1)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=45)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=45)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

# 2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout

model = Sequential()
model.add(Dense(128, activation='relu', input_dim=13))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto')

model.fit(x_train, y_train, epochs=2000, batch_size=8, validation_data=(x_val, y_val), verbose=2, callbacks=[early_stopping])

# 4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test)
print('loss :', loss)
print('MAE :', mae)

y_predict = model.predict(x_test)

# RMSE, r2
from sklearn.metrics import mean_squared_error, r2_score
def rmse(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE :', rmse(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print('R2 :', r2)

# Dropout 이전
# loss : 9.592035293579102
# MAE : 2.2911343574523926
# RMSE : 3.0971010407447044
# R2 : 0.9112489397327269

# Dropout 후
# loss : 9.605646133422852
# MAE : 2.4262447357177734
# RMSE : 3.099297468080985
# R2 : 0.9111230126963046