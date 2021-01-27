import numpy as np
import pandas as pd

submission = pd.DataFrame(np.zeros((7776,9)))
submission.columns = pd.read_csv('./dacon/data/sample_submission.csv', index_col=0, header=0).columns
submission.index = pd.read_csv('./dacon/data/sample_submission.csv', index_col=0, header=0).index

# 데이터
train_data = pd.read_csv('./dacon/data/train/train.csv', header=0)

day = 7

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

def split_to_seq(data):
    temp = []
    for i in range(48):
        temp1 = pd.DataFrame()
        for j in range(int(len(data)/48)):
            temp2 = data.iloc[j*48+i,:]
            temp2 = temp2.to_numpy()
            temp2 = temp2.reshape(1, temp2.shape[0])
            temp2 = pd.DataFrame(temp2)
            temp1 = pd.concat([temp1,temp2])
        x = temp1.to_numpy()
        temp.append(x)
    return np.array(temp)

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
    temp = temp[['TARGET', 'GHI', 'DHI', 'DNI', 'RH', 'T-Td']]

    if is_train == True:          
        temp['Target1'] = temp['TARGET'].shift(-48).fillna(method='ffill')
        temp['Target2'] = temp['TARGET'].shift(-48*2).fillna(method='ffill')
        temp = temp.dropna()
        return temp.iloc[:-96]

    elif is_train == False:
        return temp.iloc[-48*day:, :]

train_data = preprocess_data(train_data)
scaler.fit(train_data.iloc[:,:-2])
train_data.iloc[:,:-2] = scaler.transform(train_data.iloc[:,:-2])
train_data = split_to_seq(train_data)
print(train_data.shape)     # (48, 1093, 8)

test_data = []
for i in range(81):
    temp = pd.read_csv("./dacon/data/test/{}.csv".format(i), header=0)
    temp = preprocess_data(temp, is_train=False)
    temp = scaler.transform(temp)
    temp = pd.DataFrame(temp)
    temp = split_to_seq(temp)
    test_data.append(temp)
test_data = np.array(test_data)
print(test_data.shape)      # (81, 48, 7, 6)

def split_xy(data,timestep):
    x, y1, y2 = [], [], []
    for i in range(len(data)):
        x_end = i + timestep
        if x_end > len(data):
            break
        temp_x = data[i:i+timestep,:-2]
        temp_y1 = data[x_end-1:x_end,-2]
        temp_y2 = data[x_end-1:x_end,-1]
        x.append(temp_x)
        y1.append(temp_y1)
        y2.append(temp_y2)
    return np.array(x), np.array(y1), np.array(y2)

x, y1, y2 = [], [], []
for i in range(48):
    temp1, temp2, temp3 = split_xy(train_data[i], day)
    x.append(temp1)
    y1.append(temp2)
    y2.append(temp3)

x = np.array(x)
y1 = np.array(y1)
y2 = np.array(y2)

print(x.shape)      # (48, 1087, 7, 6)
print(y1.shape)     # (48, 1087, 1)
print(y2.shape)     # (48, 1087, 1)

# 모델링 
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Conv2D, MaxPooling1D, Flatten, Dropout, Reshape, Input, Concatenate

def my_model():
    input1 = Input(shape=(x.shape[2],x.shape[3]))
    layer1 = LSTM(256,activation='relu')(input1)   # swish
    layer1 = Dense(128,activation='relu')(layer1)
    layer1 = Dense(64,activation='relu')(layer1)
    layer1 = Dense(32,activation='relu')(layer1)
    layer1 = Dense(16,activation='relu')(layer1)
    layer1 = Dense(8,activation='relu')(layer1)
    layer1 = Dense(4,activation='relu')(layer1)
    output1 = Dense(1)(layer1)

    model = Model(inputs=input1,outputs=output1)
    return model

# model.summary()

from tensorflow.keras.backend import mean, maximum
def quantile_loss(q, y_true, y_pred):   # loss function
    err = (y_true - y_pred)
    return mean(maximum(q*err, (q-1)*err), axis=-1)

quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', patience=20, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.25, verbose=1)

for a in range(48):
    x_train, x_val, y1_train, y1_val, y2_train, y2_val = train_test_split(x[a], y1[a], y2[a], train_size=0.2, shuffle=False, random_state=0)
    print('{}\'s loop Start'.format(a))

    for i, j in enumerate(quantiles): # Day 7
        print('Quantile-{} fitting Start'.format(j))
        model = my_model()
        model.compile(loss=lambda y_true,y_pred : quantile_loss(j,y_true,y_pred), optimizer='adam', metrics=[lambda y_true,y_pred : quantile_loss(j,y_true,y_pred)])
        # modelpath = "./dacon/data/sunlight_model_day7_{}_qauntile{}.hdf5".format(i+1,j)
        modelpath = "../data/modelcheckpoint/sunlight_model_day7_{}_qauntile{}.hdf5".format(i+1,j)
        cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
        model.fit(x_train, y1_train, epochs=500, batch_size=256, validation_data=(x_val, y1_val), verbose=2, callbacks=[es,cp,reduce_lr])

        x_day7 = []
        for k in range(81):
            x_day7.append(test_data[k,a])
        x_day7 = np.array(x_day7)
        df_temp1 = pd.DataFrame(model.predict(x_day7).round(2))
        df_temp1[df_temp1 < 0] = 0
        num_temp1 = df_temp1.to_numpy()

        if a % 2 == 0:
            submission.loc[submission.index.str.contains(f"Day7_{int(a/2)}h00m"), [f"q_{j:.1f}"]] = num_temp1
        else:
            submission.loc[submission.index.str.contains(f"Day7_{int(a/2)}h30m"), [f"q_{j:.1f}"]] = num_temp1
        print('Quantile-{} fitting End'.format(j))
    
    for i, j in enumerate(quantiles): # Day 8
        print('Quantile-{} fitting Start'.format(j))
        model = my_model()
        model.compile(loss=lambda y_true,y_pred : quantile_loss(j,y_true,y_pred), optimizer='adam', metrics=[lambda y_true,y_pred : quantile_loss(j,y_true,y_pred)])
        # modelpath = "./dacon/data/sunlight_model_day8_{}_qauntile{}.hdf5".format(i+1,j)
        modelpath = "../data/modelcheckpoint/sunlight_model_day8_{}_qauntile{}.hdf5".format(i+1,j)
        cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
        model.fit(x_train, y2_train, epochs=500, batch_size=256, validation_data=(x_val, y1_val), verbose=2, callbacks=[es,cp,reduce_lr])

        x_day8 = []
        for k in range(81):
            x_day8.append(test_data[k,a])
        x_day8 = np.array(x_day8)
        df_temp2 = pd.DataFrame(model.predict(x_day8).round(2))
        df_temp2[df_temp2 < 0] = 0
        num_temp2 = df_temp2.to_numpy()

        if a % 2 == 0:
            submission.loc[submission.index.str.contains(f"Day8_{int(a/2)}h00m"), [f"q_{j:.1f}"]] = num_temp2
        else:
            submission.loc[submission.index.str.contains(f"Day8_{int(a/2)}h30m"), [f"q_{j:.1f}"]] = num_temp2

        print('Quantile-{} fitting End'.format(j))

    print('{}\'s loop End'.format(a))
    
print(submission)
print(submission.shape)

submission.to_csv('./dacon/data/sample_submission_final_version3.csv')

# 01:48 ~ 11:27