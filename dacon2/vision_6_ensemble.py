import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from keras.optimizers import Adam
from tensorflow_addons.optimizers import RectifiedAdam, Lookahead

# 1. 데이터

# Resize-Image
image_size = [28, 28]
resized_image_size = [256, 256]

tr_data = pd.read_csv('./dacon2/data/train.csv', index_col=0).values
ts_data = pd.read_csv("./dacon2/data/test.csv", index_col = 0).values

print(tr_data.shape)
print(ts_data.shape)

# Train_data
tr_X = tf.convert_to_tensor(tr_data[:, 2:], dtype = tf.float32)
tr_Y = tf.squeeze(tf.convert_to_tensor(tr_data[:, 0], dtype = tf.int32))

resize = tf.reshape(tr_X, (-1, 28, 28, 1))
resize_train = tf.keras.layers.experimental.preprocessing.Resizing(64, 64)(resize)

x_train = resize_train.numpy()
y_train = to_categorical(tr_Y.numpy())

print(x_train.shape)
print(y_train.shape)

# Test_Data
ts_X = tf.convert_to_tensor(ts_data[:int(len(ts_data)/10)][:,1:], dtype = tf.float32)
resize = tf.reshape(ts_X, (-1, 28, 28, 1))
resize_test = tf.keras.layers.experimental.preprocessing.Resizing(64, 64)(resize)

print(resize_test.shape)
print(type(resize_test))
print(type(resize_test.numpy()))

x_test = resize_test.numpy()

for i in range(1,10):
    ts_X_ = tf.convert_to_tensor(ts_data[int(len(ts_data)/10) * i:int(len(ts_data)/10) * (i+1)][:,1:], dtype = tf.float32)
    resize = tf.reshape(ts_X_, (-1, 28, 28, 1))
    resize_test = tf.keras.layers.experimental.preprocessing.Resizing(64, 64)(resize)

    x_test = np.append(x_test, resize_test.numpy(), axis=0)

print(type(x_test))
print(x_test.shape)

datagen = ImageDataGenerator(
    width_shift_range=(-1,1),  
    height_shift_range=(-1,1))

datagen2 = ImageDataGenerator()

steps = 40
skfold = StratifiedKFold(n_splits=steps, random_state=42, shuffle=True)

def cnn_model(x_train):
    inputs = Input(shape=x_train.shape[1:])
    layer = Conv2D(16, 3, padding='same', strides=1, activation='relu')(inputs)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.3)(layer)

    layer = Conv2D(32, 3, padding='same', strides=1, activation='relu')(layer)
    layer = BatchNormalization()(layer)
    layer = Conv2D(32, 5, padding='same', strides=1, activation='relu')(layer)
    layer = BatchNormalization()(layer)    
    layer = Conv2D(32, 5, padding='same', strides=1, activation='relu')(layer)
    layer = BatchNormalization()(layer)
    layer = Conv2D(32, 5, padding='same', strides=1, activation='relu')(layer)
    layer = BatchNormalization()(layer)
    layer = MaxPooling2D(3)(layer)
    layer = Dropout(0.3)(layer)

    layer = Conv2D(64, 3, padding='same', strides=1, activation='relu')(layer)
    layer = BatchNormalization()(layer)
    layer = Conv2D(64, 5, padding='same', strides=1, activation='relu')(layer)
    layer = BatchNormalization()(layer)
    layer = MaxPooling2D(3)(layer)
    layer = Dropout(0.3)(layer)

    layer = Flatten()(layer)

    layer = Dense(128, activation='relu')(layer)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.3)(layer)
    layer = Dense(64, activation='relu')(layer)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.3)(layer)

    outputs = Dense(10, activation='softmax')(layer)

    model = Model(inputs=inputs, outputs=outputs)

    return model

def get_opt(init_lr = 3e-3):
    radam = tfa.optimizers.RectifiedAdam(
        lr = init_lr, warmup_proportion = 0, min_lr = 1e-5, weight_decay = 1e-4)
    ranger = tfa.optimizers.Lookahead(radam)

    return ranger

val_acc = []
for i, (train_idx, val_idx) in enumerate(skfold.split(x_train, y_train.argmax(1))):
    x_train_, x_val_ = x_train[train_idx], x_train[val_idx]
    y_train_, y_val_ = y_train[train_idx], y_train[val_idx]
    
    model = cnn_model(x_train)

    filepath = './dacon2/data/vision_model_{}.hdf5'.format(i)
    es = EarlyStopping(monitor='val_loss', patience=160, mode='auto')
    cp = ModelCheckpoint(filepath=filepath, monitor='val_loss', save_best_only=True, mode='auto')
    lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=100)

    model.compile(loss='categorical_crossentropy', optimizer=get_opt(), metrics=['accuracy'])
    hist = model.fit_generator(datagen.flow(x_train_, y_train_, batch_size=32), epochs=2000,
               validation_data=(datagen.flow(x_val_, y_val_)), verbose=2, callbacks=[es, cp, lr])

    val_acc.append(max(hist.history['val_accuracy']))

    print('{}\'s CV End'.format(i+1))

# 3. 예측

# best model select
print(val_acc)

i_max = np.argmax(val_acc)

print('Best Model is {}\'s'.format(i_max))
model = load_model('./dacon2/data/vision_model_{}.hdf5'.format(i_max))

submission = pd.read_csv('./dacon2/data/submission.csv', index_col=0, header=0)

submission['digit'] = np.argmax(model.predict(x_test), axis=1)
print(submission)

submission.to_csv('./dacon2/data/submission_model_best.csv')


# KFold 값 평균내기
submission2 = pd.read_csv('./dacon2/data/submission.csv', index_col=0, header=0)

result = 0
for i in range(steps):
    model = load_model('./dacon2/data/vision_model_{}.hdf5'.format(i))

    result += model.predict_generator(datagen2.flow(x_test, shuffle=False)) / steps

submission2['digit'] = result.argmax(1)

print(submission2)
submission2.to_csv('./dacon2/data/submission_model_mean.csv')
