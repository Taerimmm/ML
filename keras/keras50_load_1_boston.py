import numpy as np

x_data = np.load('../data/npy/boston_x.npy')
y_data = np.load('../data/npy/boston_y.npy')

print(x_data.shape, y_data.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=45)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=45)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape=(13,))
dense1 = Dense(128, activation='relu')(input1)
dense2 = Dense(64, activation='relu')(dense1)
dense3 = Dense(64, activation='relu')(dense2)
dense4 = Dense(64, activation='relu')(dense3)
dense5 = Dense(64, activation='relu')(dense4)
output1 = Dense(1)(dense5)

model = Model(inputs=input1, outputs=output1)

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=20, mode='auto')
model.fit(x_train, y_train, epochs=4000, batch_size=8, validation_data=(x_val, y_val), verbose=2, callbacks=[es])

loss, mae = model.evaluate(x_test, y_test)
print('loss :', loss)
print('MAE :', mae)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score
def rmse(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE :', rmse(y_test, y_predict))
r2 = r2_score(y_test, y_predict)
print('R2 :', r2)

# Result
# loss : 12.888792991638184
# MAE : 2.693161725997925
# RMSE : 3.5900966129801355
# R2 : 0.8807454181824746