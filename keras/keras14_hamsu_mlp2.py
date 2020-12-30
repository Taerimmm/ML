# 다:다 mlp 함수형
# keras10_mlp3.py를 함수형으로 바꾸시오

import numpy as np

x = np.array([range(100), range(301,401), range(501,601), range(400, 500)]).T
y = np.array([range(101,201), range(301,401), range(501,601)]).T

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=45)


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

inputs = Input(shape=(4,))
dense1 = Dense(10, activation='relu')(inputs)
dense2 = Dense(5)(dense1)
dense3 = Dense(4)(dense2)
dense4 = Dense(5)(dense3)
dense5 = Dense(3)(dense4)
outputs = Dense(3)(dense5)
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

x_predict2 = np.array([50, 350, 550, 450]).reshape(1,4)
y_predict2 = model.predict(x_predict2)
print(y_predict2)
# [[150.61954 350.79092 550.1859 ]]