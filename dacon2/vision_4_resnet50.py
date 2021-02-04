import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, Activation
from tensorflow.keras.layers import GlobalAveragePooling2D, ZeroPadding2D, Add
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from tensorflow.keras.optimizers import Adam

# 1. 데이터
train_data = pd.read_csv("./dacon2/data/train.csv", index_col=0, header=0)
print(train_data)

'''
# 그림 확인
idx = 999
img = train_data.loc[idx, '0':].values.reshape(28, 28).astype(int)
digit = train_data.loc[idx, 'digit']
letter = train_data.loc[idx, 'letter']
plt.title('Index: %i, Digit: %s, Letter: %s'%(idx, digit, letter))
plt.imshow(img)
plt.show()
'''

train_digit = train_data['digit'].values
train_letter = train_data['letter'].values

x_train = train_data.drop(['digit', 'letter'], axis=1).values
x_train = x_train.reshape(-1, 28, 28, 1)
x_train = x_train/255

y = train_data['digit']
y_train = np.zeros((len(y), len(y.unique())))
for i, digit in enumerate(y):
    y_train[i, digit] = 1

print(x_train.shape, y_train.shape)     # (2048, 28, 28, 1) (2048, 10)

# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, stratify=y_train)

datagen = ImageDataGenerator(
    width_shift_range=(-1,1),  
    height_shift_range=(-1,1))

datagen2 = ImageDataGenerator()

steps = 40
skfold = StratifiedKFold(n_splits=steps, random_state=42, shuffle=True)

# 2. 모델
# number of classes
K = 10

input_tensor = Input(shape=x_train.shape[1:])

def conv1_layer(x):
    x = ZeroPadding2D(padding=(3,3))(x)
    x = Conv2D(64, (7,7), strides=(2,2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1,1))(x)

    return x

def conv2_layer(x):
    x = MaxPooling2D((3,3),2)(x)

    shortcut = x

    for i in range(3):
        if (i == 0):
            x = Conv2D(64, (1,1), strides=(1,1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(64, (3,3), strides=(1,1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(256, (1,1), strides=(1,1), padding='valid')(x)
            shortcut = Conv2D(256, (1,1), strides=(1,1), padding='valid')(shortcut)

            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x
        
        else:
            x = Conv2D(64, (1,1), strides=(1,1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(64, (3,3), strides=(1,1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(256, (1,1), strides=(1,1), padding='valid')(x)
            x = BatchNormalization()(x)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

    return x

def conv3_layer(x):
    shortcut = x

    for i in range(4):
        if (i == 0):
            x = Conv2D(128, (1,1), strides=(2,2), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(128, (3,3), strides=(1,1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(512, (1,1), strides=(1,1), padding='valid')(x)
            shortcut = Conv2D(512, (1,1), strides=(2,2), padding='valid')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

        else:
            x = Conv2D(128, (1,1), strides=(1,1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(128, (3,3), strides=(1,1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(512, (1,1), strides=(1,1), padding='valid')(x)
            x = BatchNormalization()(x)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

    return x

def conv4_layer(x):
    shortcut = x

    for i in range(6):
        if (i == 0):
            x = Conv2D(256, (1,1), strides=(2,2), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(256, (3,3), strides=(1,1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(1024, (1,1), strides=(1,1), padding='valid')(x)
            shortcut = Conv2D(1024, (1,1), strides=(2,2), padding='valid')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

        else:
            x = Conv2D(256, (1,1), strides=(1,1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(256, (3,3), strides=(1,1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(1024, (1,1), strides=(1,1), padding='valid')(x)
            x = BatchNormalization()(x)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

    return x

def conv5_layer(x):
    shortcut = x

    for i in range(3):
        if (i == 0):
            x = Conv2D(512, (1,1), strides=(2,2), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(512, (3,3), strides=(1,1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(2048, (1,1), strides=(1,1), padding='valid')(x)
            shortcut = Conv2D(2048, (1,1), strides=(2,2), padding='valid')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

        else:
            x = Conv2D(512, (1,1), strides=(1,1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(512, (3,3), strides=(1,1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(2048, (1,1), strides=(1,1), padding='valid')(x)
            x = BatchNormalization()(x)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

    return x
    
x = conv1_layer(input_tensor) 
x = conv2_layer(x)
x = conv3_layer(x)
x = conv4_layer(x)
x = conv5_layer(x)

# error 잡기

x = GlobalAveragePooling2D()(x)
output_tensor = Dense(K, activation='softmax')(x)

resnet50 = Model(inputs=input_tensor, outputs=output_tensor)

resnet50.summary()

val_acc = []
for i, (train_idx, val_idx) in enumerate(skfold.split(x_train, y_train.argmax(1))):
    x_train_, x_val_ = x_train[train_idx], x_train[val_idx]
    y_train_, y_val_ = y_train[train_idx], y_train[val_idx]
    
    model = resnet50(x_train)

    filepath = './dacon2/data/vision_model_{}.hdf5'.format(i)
    es = EarlyStopping(monitor='val_loss', patience=160, mode='auto')
    cp = ModelCheckpoint(filepath=filepath, monitor='val_loss', save_best_only=True, mode='auto')
    lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=100)

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.002, epsilon=None), metrics=['accuracy'])
    hist = model.fit_generator(datagen.flow(x_train_, y_train_, batch_size=32), epochs=2000,
               validation_data=(datagen.flow(x_val_, y_val_)), verbose=2, callbacks=[es, cp, lr])

    val_acc.append(max(hist.history['val_accuracy']))

    print('{}\'s CV End'.format(i+1))

# 3. 예측
test_data = pd.read_csv('./dacon2/data/test.csv', index_col=0, header=0)
x_test = test_data.drop(['letter'], axis=1).values
x_test = x_test.reshape(-1, 28, 28, 1)
x_test = x_test/255

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
