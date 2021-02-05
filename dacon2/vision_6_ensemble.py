import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import shuffle

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from keras.optimizers import Adam

# 1. 데이터
train = pd.read_csv('./dacon2/data/train.csv', index_col=0)
test = pd.read_csv('./dacon2/data/test.csv', index_col=0)

def get_tr_data(SEED):
    tr_data = train.values
    tr_data = shuffle(tr_data, random_state=SEED)

    tr_X = tf.convert_to_tensor(tr_data[:, 2:], dtype=tf.float32)
    tr_Y = tf.squeeze(tf.convert_to_tensor(tr_data[:, 0], dtype=tf.float32))

    return tr_X, tr_Y

def get_ts_data():
    ts_data = test.values
    ts_X = tf.convert_to_tensor(ts_data[:, 1:], dtype=tf.float32)

    return ts_X

image_size = [28, 28]
resized_image_size = [256, 256]

tr_data = pd.read_csv('./dacon2/data/train.csv', index_col=0).values
ts_data = pd.read_csv("./dacon2/data/test.csv", index_col = 0).values

print(tr_data.shape)
print(ts_data.shape)

tr_X = tf.convert_to_tensor(tr_data[:, 2:], dtype = tf.float32)
tr_Y = tf.squeeze(tf.convert_to_tensor(tr_data[:, 0], dtype = tf.int32))

resize = tf.reshape(tr_X, (-1, 28, 28, 1))
resize_train = tf.keras.layers.experimental.preprocessing.Resizing(256, 256)(resize)

print(resize_train)
print(resize_train.shape)

# print(ts_data[:int(len(ts_data)/10)].shape)

# ts_X = tf.convert_to_tensor(ts_data[:, 2:], dtype = tf.float32)
# resize = tf.reshape(ts_X[:int(len(ts_X)/10)], (-1, 28, 28, 1))
# resize_test = tf.keras.layers.experimental.preprocessing.Resizing(256, 256)(resize)

# # print(resize_test)
# print(resize_test.shape)
