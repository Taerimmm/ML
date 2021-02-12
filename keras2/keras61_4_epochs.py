# epochs 100적용
# validation_split, callback 적용
# early_stopping 5 적용
# Reduce LR 3 적용

# modelcheckpoint 폴더에 hdf5 파일 적용

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 1. 데이터 / 전처리
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28*28).astype('float32') / 255.
x_test = x_test.reshape(10000, 28*28).astype('float32') / 255.

# 2. 모델
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
    return {"batch_size":batches, "optimizer":optimizers, "drop":dropout}
hyperparameters = create_hyperparameters()
model2 = build_model()

# searchCV에 넣기위해 keras 모델을 감싸는 역할
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model2 = KerasClassifier(build_fn=build_model, verbose=1, epochs=3)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = RandomizedSearchCV(model2, hyperparameters, cv=3)
# search = GridSearchCV(model2, hyperparameters, cv=3)

filepath = '../data/modelcheckpoint/k61_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', patience=5, mode='auto')
cp = ModelCheckpoint(filepath=filepath, monitor='val_loss', save_best_only=True, mode='auto')
lr = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, mode='auto')
search.fit(x_train, y_train, validation_split=0.2,verbose=1, epochs=100, callbacks=[es, lr])

print(search.best_params_)
print(search.best_estimator_)
print(search.best_score_) # 밑의 score 랑 다르다.
acc = search.score(x_test, y_test)
print('최종 acc :', acc)