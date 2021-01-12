import numpy as np

x_train_data = np.load('../data/npy/fashion_x_train.npy')
x_test_data = np.load('../data/npy/fashion_x_test.npy')
y_train_data = np.load('../data/npy/fashion_y_train.npy')
y_test_data = np.load('../data/npy/fashion_y_test.npy')

print(x_train_data.shape, y_train_data.shape)
print(x_test_data.shape, y_test_data.shape)

x_train = x_train_data/255.
x_test =  x_test_data/255.

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train_data)
y_test = to_categorical(y_test_data)

print(y_train.shape, y_test.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

model = Sequential()
model.add(Conv2D(filters=256, kernel_size=(2,2), padding='same', strides=1, input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv2D(filters=128, kernel_size=(2,2), padding='same', strides=1, input_shape=(14,14,1)))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=20, mode='auto')
model.fit(x_train, y_train, epochs=30, batch_size=32, validation_split=0.2, verbose=2, callbacks=[es])

loss, acc = model.evaluate(x_test, y_test)
print('loss :', loss)
print('ACC :', acc)

# Result
# loss : 0.5638763308525085
# ACC : 0.9075999855995178