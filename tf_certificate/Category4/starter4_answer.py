# ======================================================================
# There are 5 questions in this exam with increasing difficulty from 1-5.
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score significantly
# less than your Category 5 question.
#
# Don't use lambda layers in your model.
# You do not need them to solve the question.
# Lambda layers are not supported by the grading infrastructure.
#
# You must use the Submit and Test button to submit your model
# at least once in this category before you finally submit your exam,
# otherwise you will score zero for this category.
# ======================================================================
#
# NLP QUESTION
#
# Build and train a classifier for the sarcasm dataset.
# The classifier should have a final layer with 1 neuron activated by sigmoid as shown.
# It will be tested against a number of sentences that the network hasn't previously seen
# and you will be scored on whether sarcasm was correctly detected in those sentences.

import json
import tensorflow as tf
import numpy as np
import urllib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def solution_model():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/sarcasm.json'
    urllib.request.urlretrieve(url, './tf_certificate/Category4/sarcasm.json')

    # DO NOT CHANGE THIS CODE OR THE TESTS MAY NOT WORK
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type='post'
    padding_type='post'
    oov_tok = "<OOV>"
    training_size = 20000

    sentences = []
    labels = []
    # YOUR CODE HERE
    with open('./tf_certificate/Category4/sarcasm.json', 'r') as f:
        datasets = json.load(f)
    for item in datasets:
        sentences.append(item['headline'])
        labels.append(item['is_sarcastic'])

    token = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    token.fit_on_texts(sentences)
    x = token.texts_to_sequences(sentences)

    pad_x = pad_sequences(x, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    labels = np.array(labels)

    x_train = pad_x[:training_size]
    x_test = pad_x[training_size:]
    y_train = labels[:training_size]
    y_test = labels[training_size:]


    model = tf.keras.Sequential([
    # YOUR CODE HERE. KEEP THIS OUTPUT LAYER INTACT OR TESTS MAY FAIL
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
        tf.keras.layers.Conv1D(256, 3, padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64),
        tf.keras.layers.Dense(16),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    es = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=30, mode='auto')
    lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=8, factor=0.8, mode='auto')
    model.fit(x_train, y_train, epochs=3000, batch_size=64, validation_split=0.2, callbacks=[es,lr])

    print(model.evaluate(x_test, y_test)[1])

    return model


# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("./tf_certificate/Category4/mymodel.h5")
