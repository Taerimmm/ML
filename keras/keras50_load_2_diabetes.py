import numpy as np

x_data = np.load('../data/npy/diabetes_x.npy')
y_data = np.load('../data/npy/diabetes_y.npy')

print(x_data.shape, y_data.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=.2, random_state=45)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=.2, random_state=45)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(10,)))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=50, mode='auto')
model.fit(x_train, y_train, epochs=10000, batch_size=8, validation_data=(x_val, y_val), verbose=2, callbacks=[es])

loss, mse = model.evaluate(x_test, y_test)
print("loss :", loss)
print('MSE :', mse)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score
def rmse(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE :', rmse(y_test, y_predict))
print('MSE :', mean_squared_error(y_test, y_predict))
print('R2 :', r2_score(y_test, y_predict))

# Result
# loss : 2399.428955078125
# MSE : 39.470680236816406
# RMSE : 48.98396709178776
# MSE : 2399.429032049346
# R2 : 0.5137254757949608