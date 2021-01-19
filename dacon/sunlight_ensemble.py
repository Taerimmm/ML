import numpy as np
import pandas as pd

# 데이터
data = pd.read_csv('./dacon/data/train/train.csv', header=0)
'''
print(data[data.TARGET==0][["Hour","Minute"]].drop_duplicates())
for i,j in np.array(data[data.TARGET==0][["Hour","Minute"]].drop_duplicates()):
    # print(i,j)
    print("Hour ",i,": Minute",j,'=', data[(data.Hour==i) & (data.Minute==j)].TARGET.sum())
'''

a = []
for i in range(len(data)):
    target = data.iloc[i,-1]
    if target > 0.0:
        a.append(1)
    else :
        a.append(0)
data['target_binary'] = a

data.set_index(['Day','Hour','Minute'], inplace=True)
print(data.shape)
print(data.head())

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

x_row_1, x_col_1 = 2*24*7, 6
y_row_1, y_col_1 = 2*24*2, 1
(x_data, y_data) = split_x(data.iloc[:,:-1], x_row_1, x_col_1, y_row_1, y_col_1)
x_row_2, x_col_2 = 2*24*7, 1
y_row_2, y_col_2 = 2*24*2, 1
(x_binary, y_binary) = split_x(data.iloc[:,-1:], x_row_1, x_col_1, y_row_1, y_col_1)

print(x_data.shape, y_data.shape)
print(x_binary.shape, y_binary.shape)

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(x_data, y_data, test_size=0.2)
x2_train, x2_test, y2_train, y2_test = train_test_split(x_binary, y_binary, test_size=0.2)

y1_train = y1_train.reshape(y1_train.shape[0], y1_train.shape[1])
y1_test = y1_test.reshape(y1_test.shape[0], y1_test.shape[1])
y2_train = y2_train.reshape(y2_train.shape[0], y2_train.shape[1])
y2_test = y2_test.reshape(y2_test.shape[0], y2_test.shape[1])
print(x1_train.shape, x1_test.shape)
print(x2_train.shape, x2_test.shape)
print(y1_train.shape, y1_test.shape)
print(y2_train.shape, y2_test.shape)

x_row_1, x_col_1 = 2*24*7, 6
y_row_1, y_col_1 = 2*24*2, 1
x_row_2, x_col_2 = 2*24*7, 1
y_row_2, y_col_2 = 2*24*2, 1
# 모델링 
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, Reshape, Input, Concatenate

input1 = Input(shape=(x_row_1,x_col_1))
layer1 = Conv1D(32,3,activation='relu',padding='same',strides=1)(input1)
layer1 = Conv1D(32,3,activation='relu',padding='same',strides=1)(layer1)
layer1 = MaxPooling1D(pool_size=2)(layer1)
layer1 = Dropout(0.2)(layer1)
layer1 = Conv1D(32,3,activation='relu',padding='same',strides=1)(input1)
layer1 = Conv1D(32,3,activation='relu',padding='same',strides=1)(layer1)
layer1 = MaxPooling1D(pool_size=2)(layer1)
layer1 = Dropout(0.2)(layer1)
layer1 = Flatten()(layer1)
layer1 = Dense(128, activation='relu')(layer1)
layer1 = Dense(128, activation='relu')(layer1)

input2 = Input(shape=(x_row_2,x_col_2))
layer2 = Conv1D(32,3,activation='relu',padding='same',strides=1)(input2)
layer2 = Conv1D(32,3,activation='relu',padding='same',strides=1)(layer2)
layer2 = MaxPooling1D(pool_size=2)(layer2)
layer2 = Flatten()(layer2)
layer2 = Dense(128, activation='relu')(layer2)

merge = Concatenate()([layer1, layer2])

output1 = Dense(10)(merge)
output1 = Dense(96)(output1)

output2 = Dense(10)(merge)
output2 = Dense(96, activation='softmax')(output2)

model = Model(inputs=[input1,input2],outputs=[output1,output2])

model.summary()


model.compile(loss=['mse','categorical_crossentropy'], optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', patience=50, mode='auto')
modelpath = "./dacon/data/model.hdf5"
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)
model.fit([x1_train,x2_train], [y1_train,y2_train], epochs=1000, batch_size=256, validation_split=0.2, verbose=2, callbacks=[es,cp,reduce_lr])
# model.fit([x1_train,x2_train], [y1_train,y2_train], epochs=10, batch_size=256, validation_split=0.2, verbose=1)

loss = model.evaluate([x1_test,x2_test], [y1_test,y2_test], batch_size=256)
print('loss :', loss)





'''
from tensorflow.keras.models import load_model
model = load_model('./dacon/data/model.hdf5')

# quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
a = []
for i in range(81):
    data = pd.read_csv("./dacon/data/test/{}.csv".format(i), header=0).set_index(['Day','Hour','Minute'])
    a.append(np.array(data))
x_test = np.array(a)
print(x_test.shape)
y_pred = model.predict(x_test)
print(y_pred)
print(y_pred.shape)
'''