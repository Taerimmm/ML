import numpy as np
a = np.array(range(1,11))

from tensorflow.keras.models import load_model
model = load_model('./model/save_keras35.h5')

model.summary()

# 완성해보시오

size = 5

# 모델을 구성하시오
def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

dataset = split_x(a,size)
x = dataset[:,:-1]
y = dataset[:,-1]
print(x.shape)
print(y.shape)

x = x.reshape(x.shape[0], x.shape[1], 1)

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=400, batch_size=1, verbose=2)

loss = model.evaluate(x,y)
print('loss :', loss)

# loss : 0.00011022217222489417

x_pred = dataset[-1,1:].reshape(1,4,1)
print(x_pred)
y_pred = model.predict(x_pred)
print('y_pred :', y_pred)

# y_pred : [[10.920193]]