from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

docs = ['너무 재밋어요', '참 최고에요', '참 잘 만든 영화에요',
        '추천하고 싶은 영화입니다.', '한 번 더 보고 싶네요', '글쎄요',
        '별로에요', '생각보다 지루해요', '연기가 어색해요',
        '재미없어요', '너무 재미없다', '참 재밋네요', '규현이가 잘 생기긴 했어요'
]

# 긍정 1, 부정 0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)

x = token.texts_to_sequences(docs)
print(x)

# 길이 맞추기 -> 0 채우기
from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding='pre', maxlen=5, truncating='pre') # padding='post' - 뒤쪽에 붙힘
print(pad_x)
print(pad_x.shape)      # (13, 5)

print(np.unique(pad_x))
print(len(np.unique(pad_x)))    

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Flatten, Conv1D, Conv2D

# Embedding Layer 를 빼고 모델 구성
model = Sequential()
# model.add(Embedding(input_dim=28, output_dim=11, input_length=5))

# 1.
# model.add(Conv2D(256, 3, padding='same', input_shape=(5,1,1)))
# model.add(Flatten())

# 2.
model.add(LSTM(32, input_shape=(5,1)))

# 3.
# model.add(Dense(32, input_shape=(5,)))


model.add(Dense(32))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()


# 1.
# pad_x = pad_x.reshape(13,5,1,1)
# 2.
pad_x = pad_x.reshape(13,5,1)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(pad_x, labels, epochs=100)

acc = model.evaluate(pad_x, labels)[1]
print(acc)
