# 실습
# VGG으로 만들어 봐!!!

import numpy as np
import pandas as pd
from tensorflow.keras.applications import vgg16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

datagen = ImageDataGenerator(
    validation_split=0.2
)

human_data = datagen.flow_from_directory(
    '../data/image/human',
    target_size=(224,224),
    class_mode='binary',
    batch_size=32,
    subset='training'
)

valid_data = datagen.flow_from_directory(
    '../data/image/human',
    target_size=(224,224),
    class_mode='binary',
    batch_size=32,
    subset='validation'
)

for i in human_data:
    print(i)
    print(i[0].shape)
    print(i[1].shape)
    test = i
    break

test = vgg16.preprocess_input(test[0])

vgg = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=test.shape[1:])

model = Sequential()
model.add(vgg)
model.add(Flatten())
model.add(Dense(256))
model.add(Dense(64))
model.add(Dense(16))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit_generator(human_data, epochs=100, validation_data=valid_data, verbose=2)

results = model.predict(test)

print(results)
print(i[1])