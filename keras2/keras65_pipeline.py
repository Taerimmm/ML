# 61번을 pipeline으로 구성

import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline, make_pipeline

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

import warnings
warnings.filterwarnings('ignore')

(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28*28).astype('float32') / 255.
x_test = x_test.reshape(10000, 28*28).astype('float32') / 255.

kfold = KFold(n_splits=5, shuffle=True)


def build_model(drop=0.5, optimizer='adam'):
    inputs = Input(shape=(28*28,), name='input')
    x = Dense(512, activation='relu', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation='relu', name='hidden2')(inputs)
    x = Dropout(drop)(x)
    x = Dense(128, activation='relu', name='hidden3')(inputs)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='outpus')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

    return model

def create_hyperparameters():
    batches = [10, 20, 30, 40, 50]
    optimizers = ['rmseprop', 'adam', 'adadelta']
    dropout = [0.1, 0.2, 0.3]
    return {"kerasclassifier__batch_size":batches, "kerasclassifier__optimizer":optimizers, "kerasclassifier__drop":dropout}
hyperparameters = create_hyperparameters()

model2 = KerasClassifier(build_fn=build_model, verbose=1)

pipe = Pipeline([('MinMax',MinMaxScaler()), ('kerasclassifier', model2)])

model = RandomizedSearchCV(pipe, hyperparameters, cv=kfold)

model.fit(x_train, y_train)

print(model.best_params_)
print(model.best_estimator_)
print(model.best_score_) # 밑의 score 랑 다르다.
acc = model.score(x_test, y_test)
print('최종 acc :', acc)
