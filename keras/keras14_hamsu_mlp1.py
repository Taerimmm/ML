# 다:1 mlp 함수형
# keras10_mlp2.py를 함수형으로 바꾸시오

import numpy as np

x = np.array([range(100), range(301,401), range(501,601)]).T
y = np.array([range(101,201)]).T

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=45)


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

inputs = Input(shape=(3,))
dense = Dense(10, activation='relu')(inputs)
dense = Dense(5)(dense)
dense = Dense(5)(dense)
dense = Dense(5)(dense)
outputs = Dense(1)(dense)
model = Model(inputs=inputs, outputs=outputs)


model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, verbose=2)

loss, mae = model.evaluate(x_test, y_test)
print('loss :', loss)
print('MAE :', mae)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error
def rmse(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE :', rmse(y_test, y_predict))
print('MSE :', mean_squared_error(y_test, y_predict))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('R2 :', r2)

x_predict2 = np.array([50, 350, 550]).reshape(1,3)
y_predict2 = model.predict(x_predict2)
print(y_predict2)
# [[150.96317]]