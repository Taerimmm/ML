from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.sequence import pad_sequences 
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Flatten, LSTM, Conv1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

num_words = 10000
(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words=num_words, test_split=0.2
)

print(x_train[0])
print(len(x_train[0]), len(x_train[11]))
print(y_train[0])
print('======================================')
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

print('뉴스기사 최대길이 :', max(len(i) for i in x_train))  # 2376
print('뉴스기사 평균길이 :', sum(map(len, x_train)) / len(x_train))  # 145.5398574927633 

# plt.hist([len(i) for i in x_train], bins=50)
# plt.show()

# y 분포
unique_elements, counts_elements = np.unique(y_train, return_counts=True)
print('y 분포 :', dict(zip(unique_elements, counts_elements)))
print('======================================')

# plt.hist(y_train, bins=40)
# plt.show()


# x 의 단어 분포
word_to_index = reuters.get_word_index()
print(word_to_index)
print(type(word_to_index))
print('======================================')

# 키와 벨류를 교체!!!
index_to_word = {}
for key, value in word_to_index.items():
    index_to_word[value] = key

print(index_to_word)
print(index_to_word[1])
print(len(index_to_word))
print(index_to_word[30979])


# x_train[0]
print(x_train[0])
print(' '.join(index_to_word[i] for i in x_train[0]))

print(max(x for x in (max(i)for i in x_train)))

'''
# y 카테고리 갯수 출력
category = np.max(y_train) + 1
print('y 카테고리 개수 :', category)    # 46

# y 의 유니크 한 값 출력
y_bunpo = np.unique(y_train)
print(y_bunpo)


# 전처리
x_train = pad_sequences(x_train, maxlen=100, padding='pre')
x_test = pad_sequences(x_test, maxlen=100, padding='pre')
print(x_train.shape, x_test.shape)

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
# print(y_train.shape, y_test.shape)

# 모델 구성
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=64, input_length=100))
model.add(LSTM(32))
model.add(Dense(46, activation='softmax'))

model.summary()

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=1)

results = model.evaluate(x_test, y_test)

print('loss :', results[0])
print('acc :', results[1])
'''
''' ------------------------------------------------------------------------------------------ '''

# input sequences 길이 맞추기
length = 300
x_train = pad_sequences(x_train, maxlen=length, padding='post', truncating='post')
x_test = pad_sequences(x_test, maxlen=length, padding='post', truncating='post')
print(x_train.shape, x_test.shape)

'''
# y -> to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape, y_test.shape)
'''

# 모델 
model = Sequential()
model.add(Embedding(input_dim=num_words, output_dim=256, input_length=length))
model.add(Conv1D(512, 3))
model.add(Flatten())
model.add(Dense(1024))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(46, activation='softmax'))

model.summary()

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(monitor='val_accuracy', patience=30, mode='auto')
lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.8, patience=8, mode='auto')
history = model.fit(x_train, y_train, epochs=1000, batch_size=128, validation_split=0.2, verbose=2, callbacks=[es,lr])

acc = model.evaluate(x_test, y_test)[1]
print('Acc :', acc)
# Acc : 0.754229724407196

epochs = range(1, len(history.history['accuracy']) + 1)
plt.plot(epochs, history.history['accuracy'])
plt.plot(epochs, history.history['val_accuracy'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
