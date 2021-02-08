import numpy as np

from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# 1. 데이터
dataset = load_iris()
x = dataset.data
y = dataset.target

y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=45)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
def dnn_model(node=32, drop=0.3, activation='relu', optimizer='adam'):
    inputs = Input(shape=(x_train.shape[1:]), name='input')
    layer = Dense(node*4, activation=activation, name='hidden1')(inputs)
    layer = Dense(node*3, activation=activation, name='hidden2')(layer)
    layer = Dense(node*2, activation=activation, name='hidden3')(layer)
    layer = Dense(node*1, activation=activation, name='hidden4')(layer)
    layer = Dropout(drop)(layer)
    layer = Dense(node*1, activation=activation, name='hidden5')(layer)
    outputs = Dense(3, activation='softmax', name='output')(layer)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

    return model

def param():
    node = [32, 64, 128, 256]
    dropout = [0.1, 0.2, 0.3, 0.4]
    activation = ['relu', 'selu', 'elu', 'swish']
    batches = [10, 20, 30, 40, 50]
    optimizer = ['adam', 'nadam', 'rmseprop', 'adadelta']
    return {"node":node, "drop":dropout, "optimizer":optimizer, "batch_size":batches, "activation":activation}

parameters = param()


model = KerasClassifier(build_fn=dnn_model, verbose=1)

search = RandomizedSearchCV(model, parameters, cv=3)
# search = GridSearchCV(model, parameters, cv=3)

filepath = '../data/modelcheckpoint/k62_iris_{epoch:02d}-{val_loss:.4f}.hdf5'
es = EarlyStopping(monitor='val_loss', patience=10, mode='auto')
cp = ModelCheckpoint(filepath=filepath, monitor='val_loss', save_best_only=True, mode='auto')
lr = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, mode='auto')
search.fit(x_train, y_train, validation_split=0.2, verbose=1, epochs=100, callbacks=[es, lr]) # cp

print(search.best_params_)
print(search.best_estimator_)
print(search.best_score_) # 밑의 score 랑 다르다.
acc = search.score(x_test, y_test)
print('최종 acc :', acc)

# {'optimizer': 'adam', 'node': 128, 'drop': 0.2, 'batch_size': 40, 'activation': 'swish'}
# search.best_score_ : 0.9750000238418579
# search.score       : 0.9333333373069763
