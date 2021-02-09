import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import cv2

from sklearn.model_selection import KFold
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Conv2D, BatchNormalization, Concatenate, ReLU
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, SpatialDropout2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# 1. 데이터
x_train = np.load('./dacon3/data/x_train_merge_1.npy')
for i in range(1,10):
    a = np.load('./dacon3/data/x_train_merge_{}.npy'.format(i+1))
    print(a.shape)
    x_train = np.append(x_train, a, axis=0)
print(x_train.shape)

x_train = x_train.reshape(50000, 256, 256, 1)
print(x_train.shape)
# (50000, 256, 256, 1)

y_train = pd.read_csv('./dacon3/data/dirty_mnist_2nd_answer.csv', index_col=0, header=0)
print(y_train.shape)
# (50000, 26)

x_test = np.load('./dacon3/data/x_test_merge.npy')
print(x_test.shape)

x_test = x_test.reshape(5000, 256, 256, 1)
print(x_test.shape)
# (5000, 256, 256, 1)

steps = 5
kfold = KFold(n_splits=steps, random_state=45, shuffle=True)

# 2. 모델

def bn_ReLU_conv2D(x, filters, kernel_size, strides = 1, padding = "same", weight_decay = 1e-4):
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(
        filters = filters, 
        kernel_size = kernel_size, 
        strides = strides,
        padding = padding,
        kernel_regularizer = tf.keras.regularizers.l2(weight_decay))(x)
    
    return x

def transition_block(x):
    x = BatchNormalization()(x)
    x = Conv2D(x.shape[-1] // 2, 1, padding = "same")(x)
    x = AveragePooling2D((2, 2), strides = 2)(x)

    return x

def dense_block(x, num_conv, growth_rate):
    for i in range(num_conv):
        residual = x
        x = bn_ReLU_conv2D(x, 4 * growth_rate, 1)
        x = bn_ReLU_conv2D(x, growth_rate, 3)
        x = Concatenate(axis = -1)([x, residual])

    return x


def cnn_model(x_train, dropout_rate=0.3, growth_rate=32):
    inputs = Input(shape=x_train.shape[1:])
    layer = Conv2D(2 * growth_rate, 7, strides = 2, padding = "same")(inputs)
    layer = MaxPooling2D((3, 3), strides = 2, padding = "same")(layer)

    for i, num_conv in enumerate([6, 12, 24, 16]):
        layer = dense_block(layer, num_conv, growth_rate)

        if i != 3: 
            layer = transition_block(layer)
            layer = SpatialDropout2D(dropout_rate)(layer)

    layer = GlobalAveragePooling2D()(layer)
    outputs = Dense(26, activation='sigmoid')(layer)

    model = Model(inputs=inputs, outputs=outputs)

    return model

# 3. Fit
for i, (train_idx, val_idx) in enumerate(kfold.split(x_train, y_train)):
    x_train_, x_val_ = x_train[train_idx], x_train[val_idx]
    y_train_, y_val_ = y_train.iloc[train_idx,:], y_train.iloc[val_idx,:]
    
    model = cnn_model(x_train)

    # model.summary()

    filepath = './dacon3/data/vision_2_model_{}.hdf5'.format(i)
    es = EarlyStopping(monitor='val_loss', patience=160, mode='auto')
    cp = ModelCheckpoint(filepath=filepath, monitor='val_loss', save_best_only=True, mode='auto')
    lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=100)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    model.fit(x_train_, y_train_, epochs=20, batch_size=32, validation_data=(x_val_, y_val_), verbose=2, callbacks=[es, cp, lr])

    print('{}\'s CV End'.format(i+1))
