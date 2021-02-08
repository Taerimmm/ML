# LSTM으로 수정

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 1. 데이터 / 전처리
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28*28, 1).astype('float32') / 255.
x_test = x_test.reshape(10000, 28*28, 1).astype('float32') / 255.

# 2. 모델
def build_model(node=64, drop=0.5, activation='relu', optimizer='adam'):
    inputs = Input(shape=(28*28,1), name='input')
    x = LSTM(node, activation='relu', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(node*4, activation='relu', name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(node*2, activation='relu', name='hidden3')(x)
    x = Dropout(drop)(x)
    x = Dense(node, activation='relu', name='hidden4')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

    return model

def create_hyperparameters():
    node = [64, 128, 256]
    activation=['relu', 'leaky_relu', 'elu', 'swish']
    batches = [10, 20, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = [0.1, 0.2, 0.3]
    return {"node":node, "batch_size":batches, "optimizer":optimizers, "drop":dropout}
hyperparameters = create_hyperparameters()
model2 = build_model()

# searchCV에 넣기위해 keras 모델을 감싸는 역할
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model2 = KerasClassifier(build_fn=build_model, verbose=1)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = RandomizedSearchCV(model2, hyperparameters, cv=3)
# search = GridSearchCV(mod el2, hyperparameters, cv=3)

search.fit(x_train, y_train, verbose=1)

print(search.best_params_)
print(search.best_estimator_)
print(search.best_score_) # 밑의 score 랑 다르다.
acc = search.score(x_test, y_test)
print('최종 acc :', acc)