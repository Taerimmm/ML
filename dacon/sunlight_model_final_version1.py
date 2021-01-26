import numpy as np
import pandas as pd

submission = pd.DataFrame(np.zeros((7776,9)))
submission.columns = pd.read_csv('./dacon/data/sample_submission.csv', index_col=0, header=0).columns
submission.index = pd.read_csv('./dacon/data/sample_submission.csv', index_col=0, header=0).index

# 데이터
train_data = pd.read_csv('./dacon/data/train/train.csv', header=0)

def add_features(data):    # add "GHI"
    data['cos'] = np.cos(np.pi/2 - np.abs(data['Hour']%12 - 6) / 6*np.pi/2)
    data.insert(3, 'GHI', data['DNI'] * data['cos'] + data['DHI'])
    data.drop(['cos'], axis=1, inplace=True)
    return data

def add_features2(data):   # add "Td", "T-Td"
    c = 243.12
    b = 17.62
    gamma = (b * (data['T']) / (c + (data['T']))) + np.log(data['RH'] / 100)
    dp = (c * gamma) / (b - gamma)
    data.insert(1, 'Td', dp)
    data.insert(1, 'T-Td', data['T'] - data['Td'])
    return data

def preprocess_data(data, is_train=True):
    data = add_features(data)
    data = add_features2(data)
    temp = data.copy()
    temp = temp[['Hour', 'TARGET', 'GHI', 'DHI', 'DNI', 'RH', 'T-Td']]

    if is_train == True:          
        temp['Target1'] = temp['TARGET'].shift(-48).fillna(method='ffill')
        temp['Target2'] = temp['TARGET'].shift(-48*2).fillna(method='ffill')
        temp = temp.dropna()
        return temp.iloc[:-96]

    elif is_train == False:
        return temp.iloc[-48:, :]

train_data = preprocess_data(train_data)
print(train_data.shape)     # (526464, 9)

x_data = train_data.iloc[:,:-2]
y1_data = train_data.iloc[:,-2:-1]
y2_data = train_data.iloc[:,-1:]

print(x_data.shape, y1_data.shape, y2_data.shape)

test_data = []
for i in range(81):
    temp = pd.read_csv("./dacon/data/test/{}.csv".format(i), header=0)
    temp = preprocess_data(temp, is_train=False)
    test_data.append(temp)
test_data = pd.concat(test_data)
# test_data = split_to_seq(test_data)

print(x_data.shape)         # (52464, 7)
print(test_data.shape)      # (3888, 7)


def split_x(data, size):
    a = []
    for i in range(data.shape[0] - size + 1):
        a.append(np.array(data.iloc[i:(i+size),:]))
    return np.array(a)

x_data = split_x(pd.DataFrame(x_data), 1)
y1_data = split_x(y1_data, 1)
y2_data = split_x(y2_data, 1)

print(x_data.shape)
print(y1_data.shape)
print(y2_data.shape)
print(test_data.shape)

y1_data = y1_data.reshape(y1_data.shape[0], y1_data.shape[2])
y2_data = y2_data.reshape(y2_data.shape[0], y2_data.shape[2])

from sklearn.model_selection import train_test_split
x_train, x_val, y1_train, y1_val, y2_train, y2_val = train_test_split(x_data, y1_data, y2_data, test_size=0.3, shuffle=False, random_state=0)
print(x_train.shape)
print(x_val.shape)

x_train_shape_0, x_train_shape_1, x_train_shape_2 = x_train.shape
x_val_shape_0, x_val_shape_1, x_val_shape_2 = x_val.shape

x_train = x_train.reshape(x_train_shape_0*x_train_shape_1,x_train_shape_2)
x_val = x_val.reshape(x_val_shape_0*x_val_shape_1,x_val_shape_2)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
test_data = scaler.transform(test_data)

x_train = x_train.reshape(x_train_shape_0,x_train_shape_1,x_train_shape_2)
x_val = x_val.reshape(x_val_shape_0,x_val_shape_1,x_val_shape_2)
test_data = test_data.reshape(int(test_data.shape[0]/x_train_shape_1),x_train_shape_1,x_train_shape_2)

print(x_train.shape, x_val.shape)
print(y1_train.shape, y1_val.shape)
print(y2_train.shape, y2_val.shape)
print(test_data.shape)

# 모델링 
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Conv1D, Conv2D, MaxPooling1D, Flatten, Dropout, Reshape, Input, Concatenate

def my_model():
    input1 = Input(shape=(x_train.shape[1],x_train.shape[2]))
    layer1 = Conv1D(256,2,activation='relu',padding='same',strides=1)(input1)   # swish
    layer1 = Conv1D(256,2,activation='relu',padding='same',strides=1)(layer1)
    layer1 = Conv1D(128,2,activation='relu',padding='same',strides=1)(layer1)
    layer1 = Conv1D(128,2,activation='relu',padding='same',strides=1)(layer1)
    layer1 = Flatten()(layer1)
    layer1 = Dense(64, activation='relu')(layer1)
    layer1 = Dense(64, activation='relu')(layer1)
    layer1 = Dense(32, activation='relu')(layer1)
    layer1 = Dense(32, activation='relu')(layer1)
    output1 = Dense(1)(layer1)

    model = Model(inputs=input1,outputs=output1)
    return model

# model.summary()

from tensorflow.keras.backend import mean, maximum
def quantile_loss(q, y_true, y_pred):   # loss function
    err = (y_true - y_pred)
    return mean(maximum(q*err, (q-1)*err), axis=-1)

quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for i, j in enumerate(quantiles): # Day 7
    print('Quantile-{} fitting Start'.format(j))
    model = my_model()
    model.compile(loss=lambda y_true,y_pred : quantile_loss(j,y_true,y_pred), optimizer='adam', metrics=[lambda y_true,y_pred : quantile_loss(j,y_true,y_pred)])
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    es = EarlyStopping(monitor='val_loss', patience=20, mode='auto')
    modelpath = "./dacon/data/sunlight_model_day7_{}_qauntile{}.hdf5".format(i+1,j)
    cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.25, verbose=1)
    model.fit(x_train, y1_train, epochs=500, batch_size=32, validation_data=(x_val, y1_val), verbose=2, callbacks=[es,cp,reduce_lr])

    y1_pred = model.predict(test_data).round(2)
    submission.iloc[:3888,i] = np.array([x if x > 0 else 0 for x in y1_pred.reshape(3888)])
    print('Quantile-{} fitting End'.format(j))

for i, j in enumerate(quantiles): # Day 8
    a = []
    print('Quantile-{} fitting Start'.format(j))
    model = my_model()
    model.compile(loss=lambda y_true,y_pred : quantile_loss(j,y_true,y_pred), optimizer='adam', metrics=[lambda y_true,y_pred : quantile_loss(j,y_true,y_pred)])
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    es = EarlyStopping(monitor='val_loss', patience=20, mode='auto')
    modelpath = "./dacon/data/sunlight_model_day8_{}_qauntile{}.hdf5".format(i+1,j)
    cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.25, verbose=1)
    model.fit(x_train, y2_train, epochs=500, batch_size=32, validation_data=(x_val, y2_val), verbose=2, callbacks=[es,cp,reduce_lr])

    y2_pred = model.predict(test_data).round(2)
    submission.iloc[3888:,i] = np.array([x if x > 0 else 0 for x in y2_pred.reshape(3888)])
    print('Quantile-{} fitting End'.format(j))

print(submission)
print(submission.shape)

submission.to_csv('./dacon/data/sample_submission_final_version1.csv')
