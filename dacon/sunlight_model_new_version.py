import numpy as np
import pandas as pd

submission = pd.DataFrame(np.zeros((7776,9)))
submission.columns = pd.read_csv('./dacon/data/sample_submission.csv', index_col=0, header=0).columns
submission.index = pd.read_csv('./dacon/data/sample_submission.csv', index_col=0, header=0).index

# 데이터
train_data = pd.read_csv('./dacon/data/train/train.csv', header=0)

def add_features(data):
    data['cos'] = np.cos(np.pi/2 - np.abs(data['Hour']%12 - 6)/6*np.pi/2)
    data.insert(3, 'GHI', data['DNI']*data['cos']+data['DHI'])
    data.drop(['cos'], axis=1, inplace=True)
    return data

def preprocess_data(data, is_train=True):
    temp = data.copy()
    temp = temp[['Hour', 'TARGET', 'GHI', 'DHI', 'DNI', 'WS', 'RH', 'T']]

    if is_train==True:          
        temp['Target1'] = temp['TARGET'].shift(-48).fillna(method='ffill')
        temp['Target2'] = temp['TARGET'].shift(-48*2).fillna(method='ffill')
        temp = temp.dropna()
        return temp.iloc[:-96]

    elif is_train==False:
        return temp.iloc[-48:, :]

train_data = preprocess_data(add_features(train_data))
print(train_data.iloc[:48])
print(train_data.shape)     # (52464, 10)

test_data = []
for i in range(81):
    temp = pd.read_csv("./dacon/data/test/{}.csv".format(i), header=0)
    temp = preprocess_data(add_features(temp), is_train=False)
    test_data.append(temp)
test_data = pd.concat(test_data)

print(test_data.iloc[:48])
print(test_data.shape)      # (3888, 8)

x_data = train_data.iloc[:,:-2]
y1_data = train_data.iloc[:,-2:-1]
y2_data = train_data.iloc[:,-1:]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_data)
x_data = scaler.transform(x_data)

test_data = scaler.transform(test_data)

def split_x(data, size):
    a = []
    for i in range(data.shape[0] - size + 1):
        a.append(np.array(data.iloc[i:(i+size),:]))
    return np.array(a)

x_data = split_x(pd.DataFrame(x_data), 24)
y1_data = split_x(y1_data, 24)
y2_data = split_x(y2_data, 24)

print(x_data.shape)
print(y1_data.shape)
print(y2_data.shape)
print(test_data.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y1_train, y1_test, y2_train, y2_test = train_test_split(x_data, y1_data, y2_data, test_size=0.2)

# 모델링 
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, Reshape, Input, Concatenate

input1 = Input(shape=(x_data.shape[1],x_data.shape[2]))
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
output1 = Dense(24)(layer1)

model = Model(inputs=input1,outputs=output1)

model.summary()

from tensorflow.keras.backend import mean, maximum
def quantile_loss(q, y, pred):
    err = (y-pred)
    return mean(maximum(q*err, (q-1)*err), axis=-1)

quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for i, j in enumerate(quantiles):
    print('Quantile-{} fitting Start'.format(j))
    model.compile(loss=lambda y,pred : quantile_loss(j, y, pred), optimizer='adam')
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    es = EarlyStopping(monitor='val_loss', patience=50, mode='auto')
    modelpath = "./dacon/data/sunlight_model_day7_{}_qauntile{}.hdf5".format(i+1,j)
    cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)
    model.fit(x_train, y1_train, epochs=1000, batch_size=64, validation_split=0.2, verbose=2, callbacks=[es,cp,reduce_lr])

    y1_pred = model.predict(np.array(test_data).reshape(162, 24, 8))
    submission.iloc[:3888,i] = np.array([x if x > 0 else 0 for x in y1_pred.reshape(3888)])

for i, j in enumerate(quantiles):
    a = []
    print('Quantile-{} fitting Start'.format(j))
    model.compile(loss=lambda y,pred : quantile_loss(j, y, pred), optimizer='adam')
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    es = EarlyStopping(monitor='val_loss', patience=50, mode='auto')
    modelpath = "./dacon/data/sunlight_model_day8_{}_qauntile{}.hdf5".format(i+1,j)
    cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)
    model.fit(x_train, y2_train, epochs=1000, batch_size=64, validation_split=0.2, verbose=2, callbacks=[es,cp,reduce_lr])

    y2_pred = model.predict(np.array(test_data).reshape(162, 24, 8))
    submission.iloc[3888:,i] = np.array([x if x > 0 else 0 for x in y2_pred.reshape(3888)])

print(submission)
print(submission.shape)

submission.to_csv('./dacon/data/sample_submission_new_version.csv')
