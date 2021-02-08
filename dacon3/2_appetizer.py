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
'''
# 노이즈 제거 fastNlMeansDenoising 사용 ?

img = cv2.imread('../Data/dirty_mnist_2nd/42672.png')

denoised_img = cv2.fastNlMeansDenoising(img, None, 30, 15, 21)

print(type(denoised_img))
print(img.shape)
print(denoised_img.shape)
# cv2.imshow('before', img)
# cv2.imshow("after", denoised_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

print(np.max(denoised_img), np.min(denoised_img))

a = []
size = 50
for i in range(size):
    img = cv2.imread('../Data/dirty_mnist_2nd/{0:05d}.png'.format(i))

    denoised_img = cv2.fastNlMeansDenoising(img, None, 30, 15, 21)

    a.append(denoised_img)

x_train = np.array(a)
x_train = x_train / 255
print(x_train)
print(type(x_train))
print(x_train.shape)

y_train = pd.read_csv('./dacon3/data/dirty_mnist_2nd_answer.csv', index_col=0, header=0)

y_train = y_train[:size]

print(y_train)
print(y_train.shape)

steps = 2
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


# 3. 훈련
def macro_soft_f1(y, y_hat):
    """Compute the macro soft F1-score as a cost.
    Average (1 - soft-F1) across all labels.
    Use probability values instead of binary predictions.
    
    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix of shape (BATCH_SIZE, N_LABELS)
        
    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    soft_f1 = 2*tp / (2*tp + fn + fp + 1e-16)
    cost = 1 - soft_f1 # reduce 1 - soft-f1 in order to increase soft-f1
    macro_cost = tf.reduce_mean(cost) # average on all labels
    
    return macro_cost

def macro_f1(y, y_hat, thresh=0.5):
    """Compute the macro F1-score on a batch of observations (average F1 across labels)
    
    Args:
        y (int32 Tensor): labels array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        thresh: probability value above which we predict positive
        
    Returns:
        macro_f1 (scalar Tensor): value of macro F1 for the batch
    """
    y_pred = tf.cast(tf.greater(y_hat, thresh), tf.float32)
    tp = tf.cast(tf.math.count_nonzero(y_pred * y, axis=0), tf.float32)
    fp = tf.cast(tf.math.count_nonzero(y_pred * (1 - y), axis=0), tf.float32)
    fn = tf.cast(tf.math.count_nonzero((1 - y_pred) * y, axis=0), tf.float32)
    f1 = 2*tp / (2*tp + fn + fp + 1e-16)
    macro_f1 = tf.reduce_mean(f1)
    return macro_f1



for i, (train_idx, val_idx) in enumerate(kfold.split(x_train, y_train)):
    x_train_, x_val_ = x_train[train_idx], x_train[val_idx]
    y_train_, y_val_ = y_train.iloc[train_idx,:], y_train.iloc[val_idx,:]
    
    model = cnn_model(x_train)

    # model.summary()

    filepath = './dacon3/data/vision_2_model_{}.hdf5'.format(i)
    es = EarlyStopping(monitor='val_loss', patience=160, mode='auto')
    cp = ModelCheckpoint(filepath=filepath, monitor='val_loss', save_best_only=True, mode='auto')
    lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=100)

    model.compile(loss=macro_soft_f1, optimizer='adam', metrics=[macro_f1])
    model.fit(x_train_, y_train_, epochs=20, batch_size=32, validation_data=(x_val_, y_val_), verbose=2, callbacks=[es, cp, lr])

    print('{}\'s CV End'.format(i+1))
'''
# 4. 예측

def macro_soft_f1(y, y_hat):
    """Compute the macro soft F1-score as a cost.
    Average (1 - soft-F1) across all labels.
    Use probability values instead of binary predictions.
    
    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix of shape (BATCH_SIZE, N_LABELS)
        
    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    soft_f1 = 2*tp / (2*tp + fn + fp + 1e-16)
    cost = 1 - soft_f1 # reduce 1 - soft-f1 in order to increase soft-f1
    macro_cost = tf.reduce_mean(cost) # average on all labels
    
    return macro_cost

def macro_f1(y, y_hat, thresh=0.5):
    """Compute the macro F1-score on a batch of observations (average F1 across labels)
    
    Args:
        y (int32 Tensor): labels array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        thresh: probability value above which we predict positive
        
    Returns:
        macro_f1 (scalar Tensor): value of macro F1 for the batch
    """
    y_pred = tf.cast(tf.greater(y_hat, thresh), tf.float32)
    tp = tf.cast(tf.math.count_nonzero(y_pred * y, axis=0), tf.float32)
    fp = tf.cast(tf.math.count_nonzero(y_pred * (1 - y), axis=0), tf.float32)
    fn = tf.cast(tf.math.count_nonzero((1 - y_pred) * y, axis=0), tf.float32)
    f1 = 2*tp / (2*tp + fn + fp + 1e-16)
    macro_f1 = tf.reduce_mean(f1)
    return macro_f1

a = []
size = 1
for i in range(size):
    img = cv2.imread('../Data/test_dirty_mnist_2nd/{0:05d}.png'.format(i+50000))

    denoised_img = cv2.fastNlMeansDenoising(img, None, 30, 15, 21)

    a.append(denoised_img)

x_test = np.array(a)
x_test = x_test / 255
print(x_test)
print(type(x_test))
print(x_test.shape)


# KFold 값 평균내기
submission = pd.read_csv('./dacon3/data/sample_submission.csv', index_col=0, header=0)
print(submission)

result = 0
steps = 2
for i in range(steps):
    model = load_model('./dacon3/data/vision_2_model_{}.hdf5'.format(i), custom_objects={'macro_soft_f1':macro_soft_f1, 'macro_f1':macro_f1})

    result += model.predict(x_test)
    print(result)

# print(submission)
# submission.to_csv('./dacon3/data/vision_2_submission.csv')
