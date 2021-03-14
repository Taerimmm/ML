import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

data = pd.read_csv('./project/mini/data/total_genres_mfcc.csv', header=0)
print(data.shape)

x = data.iloc[:,1:-1]

label_dict = {
    'hiphop':0,
    'rock':1,
    'pop':2,
    'folk':3,
    'electronic':4,
    'jazz':5,
    'blues':6,
    'classical':7,
    'reggae':8,
    'disco':9,
    'country':10,
    'ballad':11,
    'dance':12
}

y = data.iloc[:,-1].map(label_dict).values
y = to_categorical(y)

print(x.shape)  # (6194, 23)
print(y.shape)  # (6194, 13)

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)

# 모델
model = Sequential()
model.add(Dense(1024, activation='relu', input_shape=x.shape[1:]))
model.add(Dense(512))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(13, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
file_path = './project/mini/data/genre_mfcc_model.hdf5'
es = EarlyStopping(monitor='val_accuracy', patience=80)
cp = ModelCheckpoint(filepath=file_path, monitor='val_accuracy', save_best_only=True)
lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.8, patience=30)
history = model.fit(x_train, y_train, epochs=5000, batch_size=32, validation_data=(x_val, y_val), verbose=2, callbacks=[es,cp,lr])


# plt
train_loss = history.history['accuracy']
test_loss = history.history['val_accuracy']

plt.figure(figsize=(12,8))
plt.plot(train_loss, label='Training accuracy', color='blue')
plt.plot(test_loss, label='Testing accuracy', color='red')

plt.title('Training and Testing Accuracy by Epoch', fontsize=25)
plt.xlabel('Epoch', fontsize=18)
plt.ylabel('Accuracy', fontsize=18)
# plt.xticks(range(1,len(train_loss), range(1,len(test_loss))))
plt.legend()
plt.show()
