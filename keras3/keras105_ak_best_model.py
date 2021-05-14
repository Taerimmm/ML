import numpy as np
import tensorflow as tf
import autokeras as ak

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32') / 256.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32') / 256.

from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = ak.ImageClassifier(
    overwrite=True,
    max_trials=1,
    loss='mse'
)

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

es = EarlyStopping(monitor='val_loss', mode='min', patience=6)
lr = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose=2)
ck = ModelCheckpoint('./keras3/', save_weights_only=True, save_best_only=True, monitor='val_loss')

model.fit(x_train, y_train, epochs=1, validation_split=0.2, callbacks=[es, lr, ck])

results = model.evaluate(x_test, y_test)
print(results)

# model.summary() 

model2 = model.export_model()
model2.save('./keras3/save/aaa.h5')

best_model = model.tuner.get_best_model()
best_model.save('./keras3/save/best_aaa.h5')

# [loss               , acc]
# [0.07032570987939835, 0.0036045669112354517]