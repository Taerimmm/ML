# 실습
# cifar10 으로 VGG16 넣어서 만들 것!!

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import VGG16, vgg16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=x_train.shape[1:])

vgg16.trainable = False

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(256))
model.add(Dense(64))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=20, mode='auto')
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=7, mode='auto')

model.fit(x_train, y_train, epochs=1000, batch_size=64, validation_data=(x_test, y_test), verbose=2, callbacks=[es, lr])

# loss: 1.0503 - accuracy: 0.6378 - val_loss: 1.1457 - val_accuracy: 0.6163
