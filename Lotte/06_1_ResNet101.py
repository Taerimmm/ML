import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

x_train = np.load('../data/LPD_competition/npy/train_data.npy')
y_train = np.load('../data/LPD_competition/npy/label_data.npy')
print(x_train.shape)
print(y_train.shape)

y_train = to_categorical(y_train)

x_test = np.load('../data/LPD_competition/npy/test_data.npy')
print(x_test.shape)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

train_generator = ImageDataGenerator(rescale=1./255, width_shift_range=0.1, height_shift_range=0.1).flow(x_train, y_train, batch_size=32)
val_generator = ImageDataGenerator(rescale=1./255, width_shift_range=0.1, height_shift_range=0.1).flow(x_val, y_val, batch_size=32)

test_generator = ImageDataGenerator(rescale=1./255).flow(x_test, shuffle=False)

resnet = ResNet101(weights='imagenet', include_top=False, input_shape=(128,128,3))
# resnet.summary()

resnet.trainable = False

input_tensor = Input(shape=(128,128,3))
layer = resnet(input_tensor)
layer = GlobalAveragePooling2D()(layer)
layer = Flatten()(layer)
layer = Dense(2048, activation='relu')(layer)
layer = Dense(1024, activation='relu')(layer)
output_tensor = Dense(1000, activation='softmax')(layer)

model = Model(inputs=input_tensor, outputs=output_tensor)

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
path = './Lotte/resnet101_model.hdf5'
es = EarlyStopping(monitor='val_accuracy', patience=30)
cp = ModelCheckpoint(path, monitor='val_accuracy', save_best_only=True)
lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.8, patience=10)

model.fit(train_generator, epochs=2000, batch_size=32, validation_data=val_generator, callbacks=[es,cp,lr])

pred = model.predict(test_generator)
print(np.argmax(pred,1))

answer = pd.read_csv('./Lotte/sample.csv', header=0)
print(answer.shape)

answer.iloc[:,1] = np.argmax(pred,1)
print(answer)
answer.to_csv('./Lotte/submission.csv', index=False)

# 20.424