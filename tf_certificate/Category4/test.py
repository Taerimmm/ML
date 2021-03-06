import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


sentences = []
labels = []

with open('./tf_certificate/Category4/sarcasm.json', 'r') as f:
    datasets = json.load(f)
for item in datasets:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])

print(sentences)
# print(labels)

token = Tokenizer(num_words=1000, oov_token='<OOV>')
token.fit_on_texts(sentences)
x = token.texts_to_sequences(sentences)
print(x)
print(len(x))

pad_x = pad_sequences(x, maxlen=120, padding='post', truncating='post')
print(pad_x)
print(pad_x.shape)

labels = np.array(labels)

x_train = pad_x[:20000]
x_test = pad_x[20000:]
y_train = labels[:20000]
y_test = labels[20000:]

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)
