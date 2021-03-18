import os
import numpy as np
import pandas as pd
import cv2

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import StratifiedKFold

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Found 39000 images belonging to 1000 classes.
train_generator = ImageDataGenerator(rescale=1./255, validation_split=0.2).flow_from_directory(
    '../data/LPD_competition/train',
    target_size=(128,128),
    # color_mode='grayscale',
    subset='training'
)
# Found 9000 images belonging to 1000 classes.
val_generator = ImageDataGenerator(rescale=1./255, validation_split=0.2).flow_from_directory(
    '../data/LPD_competition/train',
    target_size=(128,128),
    # color_mode='grayscale',
    subset='validation'
)

print(train_generator)
print(val_generator)

# Found 72000 images belonging to 1 classes.
test_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
    '../data/LPD_competition',
    target_size=(128,128),
    # color_mode='grayscale',
    classes=['test'],
    shuffle=False,
    class_mode=None
)

print(test_generator)

mobilenetv2 = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128,128,3))

mobilenetv2.trainable = False

input_tensor = Input(shape=(128,128,3))
layer = mobilenetv2(input_tensor)
layer = GlobalAveragePooling2D()(layer)
layer = Flatten()(layer)
layer = Dense(1024, activation='relu')(layer)
layer = Dense(1024, activation='relu')(layer)
layer = Dense(512, activation='relu')(layer)
output_tensor = Dense(1000, activation='softmax')(layer)

model = Model(inputs=input_tensor, outputs=output_tensor)

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
path = './Lotte/model.hdf5'
es = EarlyStopping(monitor='val_accuracy', patience=30)
cp = ModelCheckpoint(path, monitor='val_accuracy', save_best_only=True)
lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.8, patience=10)
 
model.fit(train_generator, epochs=2000, batch_size=32, validation_data=val_generator, callbacks=[es,cp,lr])

pred = model.predict(test_generator)
print(np.argmax(pred,1))

answer = pd.read_csv('./Lotte/sample.csv', header=0)

answer.iloc[:,1] = np.argmax(pred,1)
print(answer)
answer.to_csv('./Lotte/submission.csv', index=False)

# 0.103