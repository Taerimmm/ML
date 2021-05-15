import numpy as np
import pandas as pd
import math
import os
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Reshape, GRU, RNN

tf.keras.backend.set_floatx('float64')

train=pd.read_csv('./dacon4/data/train.csv', encoding='cp949')
test=pd.read_csv('./dacon4/data/test.csv', encoding='cp949')
submission=pd.read_csv('./dacon4/data/sample_submission.csv', encoding='cp949')

# print(train)
# print(test)
# print(submission)

mini = train.iloc[:,2].min()
size = train.iloc[:,2].max() - train.iloc[:,2].min()
train.iloc[:,2] = (train.iloc[:,2] - mini) / size

# Parameter
input_window = 996 # 임의의 수
output_window = 24 # 168 7일 24시간
window = 12 # window는 12시간 마다
num_features = 1
num_power = 60
end_ = 168
lstm_units = 32
dropout = 0.2
EPOCH = 30
BATCH_SIZE = 128


train_x = tf.reshape(train.iloc[:,2].values, [num_power, 24*85, num_features])
print(f'train_x.shape : {train_x.shape}')

train_window_x = np.zeros((train_x.shape[0], (train_x.shape[1] - (input_window + output_window))//window, input_window, num_features))
train_window_y = np.zeros((train_x.shape[0], (train_x.shape[1] - (input_window + output_window))//window, output_window, num_features))
print(f'train_window_x.shape : {train_window_x.shape}')
print(f'train_window_y.shape : {train_window_y.shape}')


# train_window_x 에 train 값 채워 넣기
for example in range(train_x.shape[0]):

    for start in range(0, train_x.shape[1] - (input_window + output_window), window):
        end = start + input_window
        train_window_x[example, start//window, :] = train_x[example, start:end, :]
        train_window_y[example, start//window, :] = train_x[example, end:end + output_window, :]

#new_train_x, reshape통해 lstm에 알맞은 형태로 집어넣기
new_train_x = tf.reshape(train_window_x, [-1, input_window, num_features])
new_train_y = tf.reshape(train_window_y, [-1, output_window, num_features])

print(f'new_train_x.shape : {new_train_x.shape}')
print(f'new_train_y.shape : {new_train_y.shape}')


# Model
model = Sequential([
    LSTM(lstm_units, return_sequences=False, recurrent_dropout=dropout),
    Dense(output_window * num_features, kernel_initializer=tf.initializers.zeros()),
    Reshape([output_window, num_features])
])

model.compile(optimizer='rmsprop', loss='mae', metrics=['mae'])

# Epo가 끝날 때 마다 점(.)을 출력해 훈련 진행 과정을 표시합니다.
class PrintDot(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 10 == 0:
            print("")
            print('.', end="")

# 가장 좋은 성능을 낸 val_loss가 적은 model만 남겨 놓았습니다.
save_best_only = tf.keras.callbacks.ModelCheckpoint(filepath='./dacon4/data/lstm_model.h5', monitor='val_loss', save_best_only=True)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

reduceLR = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10)


model.fit(new_train_x, new_train_y, epochs=EPOCH, batch_size=BATCH_SIZE, validation_split=0.2, verbose=0, callbacks=[PrintDot(), early_stop, save_best_only, reduceLR])

model.summary()


prediction = np.zeros((num_power, end_, num_features))
new_test_x = train_x

for i in range(end_//output_window):
    start_ = i * output_window
    next_ = model.predict(new_test_x[:, -input_window:, :])
    new_test_x = tf.concat([new_test_x, next_], axis=1)
    print(new_test_x.shape)
    prediction[:, start_:start_ + output_window, :] = next_
prediction = prediction * size + mini


submission['answer'] = prediction.reshape([-1, 1])
submission.to_csv('./dacon4/data/baseline_submission1.csv', index=False)