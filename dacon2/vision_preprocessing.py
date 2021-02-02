import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

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

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=45)

datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1)

train_generator = datagen.flow(x_train, y_train)

# 2. 모델
def cnn_model(x_train):
    inputs = Input(shape=x_train.shape[1:])
    layer = BatchNormalization()(inputs)
    layer = Conv2D(128, 5, padding='same', strides=1, activation='relu')(layer)
    layer = BatchNormalization()(layer)
    layer = Conv2D(128, 2, padding='same', strides=1, activation='relu')(layer)
    layer = MaxPooling2D(pool_size=2)(layer)
    layer = BatchNormalization()(layer)
    layer = Conv2D(256, 2, padding='same', strides=1, activation='relu')(layer)
    layer = BatchNormalization()(layer)
    layer = Conv2D(256, 2, padding='same', strides=1, activation='relu')(layer)
    layer = MaxPooling2D(pool_size=2)(layer)
    layer = Flatten()(layer)
    layer = BatchNormalization()(layer)
    layer = Dense(1000, activation='relu')(layer)
    layer = BatchNormalization()(layer)
    outputs = Dense(10, activation='softmax')(layer)

    model = Model(inputs=inputs, outputs=outputs)

    return model

model = cnn_model(x_train)

filepath = '../dacon2/data/vision_model.hdf5'
es = EarlyStopping(monitor='val_loss', patience=20, mode='auto')
cp = ModelCheckpoint(filepath=filepath, monitor='val_loss', save_best_only=True, mode='auto')
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=10)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit_generator(datagen.flow(x_train, y_train, batch_size=32), epochs=200, steps_per_epoch=x_train.shape[0]//32,
    validation_data=(x_val, y_val), verbose=2, callbacks=[es, cp, lr])

# 3. 예측
test_data = pd.read_csv('./dacon2/data/test.csv', index_col=0, header=0)
x_test = test_data.drop(['letter'], axis=1).values
x_test = x_test.reshape(-1, 28, 28, 1)
x_test = x_test/255

submission = pd.read_csv('./dacon2/data/submission.csv', index_col=0, header=0)

submission['digit'] = np.argmax(model.predict(x_test), axis=1)
print(submission)

submission.to_csv('./dacon2/data/submission_test.csv')
















'''
# PCA
from sklearn.decomposition import PCA

x = train_data.iloc[:,2:]
y = train_data.iloc[:,:2]
print(x.shape, y.shape)

pca = PCA()
x = pca.fit_transform(x)

cumsum = np.cumsum(pca.explained_variance_ratio_)
print(cumsum)

d = np.argmax(cumsum >= 0.999) + 1
print('d :', d)

pca = PCA(n_components=d)
x = pca.fit_transform(x)

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=45)
'''
