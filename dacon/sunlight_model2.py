import numpy as np
import pandas as pd

# 데이터
data = pd.read_csv('./dacon/data/train/train.csv', header=0)

data.set_index(['Day','Hour','Minute'], inplace=True)
print(data.shape)
print(data.head())

a = []
for i in range(len(data)):
    target = data.iloc[i,-1]
    if target > 0.0:
        a.append(1)
    else :
        a.append(0)
data['target_binary'] = a

def split_x(data, x_row, x_col, y_row, y_col):
    a, b =[], []
    x_step, y_step = 0, 0
    for i in range(data.shape[0] - x_row + 1):
        x_step += 1
        a.append(np.array(data.iloc[i:i+x_row,:x_col]))

    for i in range(data.shape[0] - y_row + 1):
        if x_row + y_row+ i > data.shape[0]:
            break
        y_step += 1
        b.append(np.array(data.iloc[x_row+i:x_row+i+y_row,-y_col:]))

    a = np.array(a)[:y_step]
    b = np.array(b)

    return  a, b

x_row, x_col = 2*24*7, 1
y_row, y_col = 2*24*2, 1
(x_data, y_data) = split_x(data.iloc[:,-2:], x_row, x_col, y_row, y_col)
print(x_data.shape)
print(y_data.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)

# 모델링 
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, Reshape, Input, Concatenate

input1 = Input(shape=(x_row,x_col))
layer1 = Conv1D(32,3,activation='relu',padding='same',strides=1)(input1)
layer1 = Conv1D(32,3,activation='relu',padding='same',strides=1)(layer1)
layer1 = MaxPooling1D(pool_size=2)(layer1)
layer1 = Dropout(0.2)(layer1)
layer1 = Conv1D(32,3,activation='relu',padding='same',strides=1)(input1)
layer1 = Conv1D(32,3,activation='relu',padding='same',strides=1)(layer1)
layer1 = MaxPooling1D(pool_size=2)(layer1)
layer1 = Dropout(0.2)(layer1)
layer1 = Flatten()(layer1)
layer1 = Dense(64, activation='relu')(layer1)
layer1 = Dense(64, activation='relu')(layer1)
output1 = Dense(96)(layer1)

model = Model(inputs=input1,outputs=output1)

model.summary()

model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', patience=50, mode='auto')
modelpath = "./dacon/data/sunlight_model_binary.hdf5"
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)
model.fit(x_train, y_train, epochs=1000, batch_size=256, validation_split=0.2, verbose=2, callbacks=[es,cp,reduce_lr])

loss = model.evaluate(x_test, y_test, batch_size=256)
print('loss :', loss)
