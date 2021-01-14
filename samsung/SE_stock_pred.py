import numpy as np

x_data = np.load('./samsung/etc/stock_x_data.npy')
x_train = np.load('./samsung/etc/stock_x_train.npy')
x_test = np.load('./samsung/etc/stock_x_test.npy')
y_train = np.load('./samsung/etc/stock_y_train.npy')
y_test = np.load('./samsung/etc/stock_y_test.npy')

from tensorflow.keras.models import load_model

model = load_model('./samsung/etc/stock_model.h5')
# model = load_model('../data/modelcheckpoint/stock_model.hdf5')

loss = model.evaluate(x_test, y_test)
print('loss :', loss)

# 2021-01-13 예측
last = x_data[-1].reshape(1,x_train.shape[1],x_train.shape[2])
a = model.predict(last)
print('\'2021-01-03\'의 종가는', a, '로 예측 됩니다.')

# '2021-01-03'의 종가는 [[88053.53]] 로 예측 됩니다.