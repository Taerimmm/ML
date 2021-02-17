import os
import numpy as np
import pandas as pd
import librosa
from tensorflow.keras.utils import to_categorical


import warnings
warnings.filterwarnings('ignore')

labels = []
mel_spec = []
for dir in os.scandir('../data/project_data/mini/genre'):
    for file in os.scandir(dir):
        # Loading in the audio file
        y, sr = librosa.core.load(file)

        # print(str(file).split('.')[0][11:])
        label = str(file).split('.')[0][11:]
        labels.append(label)
        
        # Computing the mel spectrograms
        spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
        spect = librosa.power_to_db(spect, ref=np.max)
        # print(spect)
        # print(spect.shape)

        # Adding the size to be 128 x 660
        if spect.shape[1] != 660:
            spect.resize(128,660, refcheck=False)
        # print(spect)
        # print(spect.shape)

        mel_spec.append(spect)

X_1 = np.array(mel_spec)
print(X_1.shape)        # (1000, 128, 660)
print('GTZAN Finish!!')

mel_spec = []
for dir in os.scandir('../data/project_data/mini/fma'):
    for file in librosa.util.find_files(dir):
        # Loading in the audio file
        y, sr = librosa.load(file)

        # print(str(file).split('.')[0][34:])
        label = str(file).split('.')[0][34:]
        labels.append(label)
        
        # Computing the mel spectrograms
        spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
        spect = librosa.power_to_db(spect, ref=np.max)
        # print(spect)
        # print(spect.shape)

        # Adding the size to be 128 x 660
        if spect.shape[1] != 660:
            spect.resize(128,660, refcheck=False)
        # print(spect)
        # print(spect.shape)

        mel_spec.append(spect)

X_2 = np.array(mel_spec)
print(X_2.shape)        # (7973, 128, 660)
print('FMA Finish!!')


X = np.concatenate((X_1, X_2))
X = X.reshape(X.shape[0], 128, 660, 1)
print(X.shape) # (8973, 128, 660, 1)

np.save('./project/mini/data/X.npy', arr=X)
print('save X!!')


label_dict = {
    'hiphop':0,
    'rock':1,
    'pop':2,
    'instrumental':3,
    'folk':4,
    'electronic':5,
    'international':6,
    'experimental':7,
    'jazz':8,
    'blues':9,
    'classical':10,
    'reggae':11,
    'disco':12,
    'metal':13,
    'country':14
}

y = pd.Series(labels).map(label_dict).values

y = to_categorical(y)
print(y.shape)  # (8973, 15)

np.save('./project/mini/data/y.npy', arr=y)
print('save y')
