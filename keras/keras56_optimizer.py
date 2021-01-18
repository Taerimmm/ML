import numpy as np

# 1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 2. 모델구성
model = Sequential()
model.add(Dense(1000, input_dim=1))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

# 3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

# optimizer = Adam(lr=0.1)
# loss : 1.2504207234087517e-06 결과물 : [[11.000145]]
# optimizer = Adam(lr=0.01)
# loss : 2.2396307898520096e-12 결과물 : [[10.999997]]
optimizer = Adam(lr=0.001)
# loss : 5.787370365925582e-13 결과물 : [[11.000002]]
# optimizer = Adam(lr=0.0001)
# loss : 7.90847661846783e-06 결과물 : [[10.996267]]

# optimizer = Adadelta(lr=0.1)
# loss : 0.0006293912301771343 결과물 : [[11.041929]]
# optimizer = Adadelta(lr=0.01)
# loss : 4.722342418972403e-05 결과물 : [[10.989835]]
# optimizer = Adadelta(lr=0.001)
# loss : 12.749245643615723 결과물 : [[4.6129127]]
# optimizer = Adadelta(lr=0.0001)
# loss : 34.84239959716797 결과물 : [[0.5278108]]

# optimizer = Adamax(lr=0.1)
# loss : 1.4242507219314575 결과물 : [[12.543798]]
# optimizer = Adamax(lr=0.01)
# loss : 5.4001247917767614e-12 결과물 : [[10.999996]]
# optimizer = Adamax(lr=0.001)
# loss : 1.84565323024799e-07 결과물 : [[10.999075]]
# optimizer = Adamax(lr=0.00001)
# loss : 0.005963744595646858 결과물 : [[10.9013815]]

# optimizer = Adagrad(lr=0.1)
# loss : 375.176513671875 결과물 : [[23.593168]]
# optimizer = Adagrad(lr=0.01)
# loss : 1.1516086001472914e-10 결과물 : [[11.000024]]
# optimizer = Adagrad(lr=0.001)
# loss : 7.479821306333179e-06 결과물 : [[10.997218]]
# optimizer = Adagrad(lr=0.0001)
# loss : 0.005406515207141638 결과물 : [[10.907624]]

# optimizer = RMSprop(lr=0.1)
# loss : 6191234048.0 결과물 : [[-140911.34]]
# optimizer = RMSprop(lr=0.01)
# loss : 20.434249877929688 결과물 : [[2.0392785]]
# optimizer = RMSprop(lr=0.001)
# loss : 0.010835723951458931 결과물 : [[10.88081]]
# optimizer = RMSprop(lr=0.0001)
# loss : 0.0008416902273893356 결과물 : [[11.052283]]

# optimizer = SGD(lr=0.1)
# loss : nan 결과물 : [[nan]]
# optimizer = SGD(lr=0.01)
# loss : nan 결과물 : [[nan]]
# optimizer = SGD(lr=0.001)
# loss : 1.0035940249508712e-05 결과물 : [[10.993355]]
# optimizer = SGD(lr=0.0001)
# loss : 0.0018033606465905905 결과물 : [[10.94672]]

# optimizer = Nadam(lr=0.1)
# loss : 0.34825849533081055 결과물 : [[10.073115]]
# optimizer = Nadam(lr=0.01)
# loss : 4941.18505859375 결과물 : [[-72.40904]]
# optimizer = Nadam(lr=0.001)
# loss : 7.531753270107605e-14 결과물 : [[11.000001]]
# optimizer = Nadam(lr=0.0001)
# loss : 5.952440915280022e-06 결과물 : [[10.99608]]

model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
model.fit(x, y, epochs=100, batch_size=1)

# 4. 평가, 예측
loss, mse = model.evaluate(x, y, batch_size=1)
y_pred = model.predict([11])
print('loss :', loss, '결과물 :', y_pred)