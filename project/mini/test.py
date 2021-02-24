import numpy as np
import tensorflow as tf

x = np.array([range(1,11), range(31,41), range(1,11), range(40,50), range(21,31)])
y = np.array([range(11, 21), range(50,60)])

print(x.shape, y.shape)

x = np.transpose(x)
y = np.transpose(y)

print(x.shape, y.shape)

input1 = tf.keras.layers.Input(shape=(5,))
x1 = tf.keras.layers.Dense(8, activation='relu')(input1)
x1 = tf.keras.layers.Dense(4, activation='relu')(x1)
out = tf.keras.layers.Dense(2)(x1)
model = tf.keras.models.Model(inputs=input1, outputs=out)

model.summary()

print(model.get_weights())
print(model.get_weights()[0].shape)
print(model.get_weights()[1].shape)
print(model.get_weights()[2].shape)
print(model.get_weights()[3].shape)
print(model.get_weights()[4].shape)
print(model.get_weights()[5].shape)

model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(x, y, epochs=1)

print(model.get_weights())
print(model.get_weights()[0].shape)
print(model.get_weights()[1].shape)
print(model.get_weights()[2].shape)
print(model.get_weights()[3].shape)
print(model.get_weights()[4].shape)
print(model.get_weights()[5].shape)