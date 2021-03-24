import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Flatten, Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau



x_train = np.load("../data/LPD_competition/npy/train_data.npy")
y_train = np.load("../data/LPD_competition/npy/label_data.npy")
x_pred = np.load('../data/LPD_competition/npy/test_data.npy')

x_train = preprocess_input(x_train)
x_pred = preprocess_input(x_pred)

y_train = to_categorical(y_train)

train_generator = ImageDataGenerator(
    width_shift_range=(-1,1),  
    height_shift_range=(-1,1), 
    zoom_range=0.15
)

val_generator = ImageDataGenerator()

print(x_train.shape)
print(y_train.shape)
print(x_pred.shape)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.9, shuffle = True, random_state=42)

train_generator = train_generator.flow(x_train, y_train, batch_size=16)
valid_generator = val_generator.flow(x_val, y_val)
test_generator = x_pred


b7 = EfficientNetB7(weights="imagenet", include_top=False, input_shape=(128,128,3))
layer = b7.output
layer = GlobalAveragePooling2D()(layer)
output = Dense(1000, activation="softmax")(layer)

model = Model(inputs=b7.input, outputs=output)

model.summary()

model.compile(optimizer=SGD(learning_rate=0.015, momentum=0.9), loss = 'categorical_crossentropy', metrics=['accuracy'])

path = './Lotte/b7_model.hdf5'
es = EarlyStopping(monitor='val_accuracy', patience=20)
cp = ModelCheckpoint(path, monitor='val_accuracy', save_best_only=True)
lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=10)

model.fit(train_generator, epochs=2000, batch_size=32, validation_data=valid_generator, callbacks=[es,cp,lr])


# predict
model.load_weights('./Lotte/b7_model.hdf5')
result = model.predict(test_generator, verbose=1)
    
answer = pd.read_csv('./Lotte/sample.csv', header=0)

answer.iloc[:,1] = np.argmax(result, 1)

answer.to_csv('./Lotte/submission.csv', index=False)
