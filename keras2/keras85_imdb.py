import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import imdb

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Flatten, LSTM, Conv1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

num_words = 10000
(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=num_words
)

# 이진 분류

print(x_train[0])
print(len(x_train[0]), len(x_train[11]))
print(y_train[0])
print('======================================')
print(x_train.shape, x_test.shape)  # (25000,) (25000,)
print(y_train.shape, y_test.shape)  # (25000,) (25000,)

max_len = 100
x_train = pad_sequences(x_train, maxlen=max_len, padding='pre', truncating='pre')
x_test = pad_sequences(x_test, maxlen=max_len, padding='pre', truncating='pre')

model = Sequential()
model.add(Embedding(input_dim=num_words, output_dim=256, input_length=max_len))
model.add(Conv1D(64, 3))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='sigmoid'))

model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(monitor='val_accuracy', patience=30, mode='auto')
lr = ReduceLROnPlateau(monitor='val_accuracy', patience=9, factor=0.8, mode='auto')
history = model.fit(x_train, y_train, epochs=100, batch_size=64, validation_split=0.2, verbose=2, callbacks=[es,lr])

acc = model.evaluate(x_test, y_test)[1]
print('Acc :', acc)

epochs = range(1, len(history.history['accuracy']) + 1)
plt.plot(epochs, history.history['accuracy'])
plt.plot(epochs, history.history['val_accuracy'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
