import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import tensorflow.keras.backend as K

def custom_mse(y_true, y_pred):
    return tf.math.reduce_mean(tf.square(y_true - y_pred))

def quantile_loss(y_true, y_pred):
    qs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    q = tf.constant(np.array([qs]), dtype=tf.float32)
    e = y_true - y_pred
    v = tf.maximum(q*e, (q-1)*e)
    return K.mean(v)


# 1. 데이터
x = np.array([1,2,3,4,5,6,7,8]).astype('float32')
y = np.array([1,2,3,4,5,6,7,8]).astype('float32')
print(x.shape)

# 2. 모델
model = Sequential()
model.add(Dense(10, input_shape=(1,)))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss=quantile_loss, optimizer='adam')

model.fit(x, y, batch_size=1, epochs=30)

loss = model.evaluate(x,y)

print(loss)

# custom_mse
# 0.007067511789500713

# quantile_loss
# 0.038379110395908356