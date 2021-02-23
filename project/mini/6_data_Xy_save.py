import os
import numpy as np
import pandas as pd
import librosa
from tensorflow.keras.utils import to_categorical

import warnings
warnings.filterwarnings('ignore')

labels = []
mel_spec = []
for dir in os.scandir('../data/project_data/mini/new_genre'):
    for file in os.scandir(dir):
        # Loading in the audio file
        y, sr = librosa.core.load(file)

        label = str(file).split('.')[0][11:]
        labels.append(label)
        
        # Computing the mel spectrograms
        spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
        spect = librosa.power_to_db(spect, ref=np.max)

        # Adding the size to be 128 x 660
        if spect.shape[1] != 660:
            spect.resize(128,660, refcheck=False)

        mel_spec.append(spect)

X_1 = np.array(mel_spec)
print(X_1.shape)        # (1200, 128, 660)
print('GTZAN Finish!!')

mel_spec = []
for dir in os.scandir('../data/project_data/mini/fma'):
    for file in librosa.util.find_files(dir):

        label = str(file).split('.')[0][43:]
        
        if label in ['experimental', 'instrumental', 'international']:
            continue

        # Loading in the audio file
        y, sr = librosa.load(file)

        labels.append(label)
        
        # Computing the mel spectrograms
        spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
        spect = librosa.power_to_db(spect, ref=np.max)

        # Adding the size to be 128 x 660
        if spect.shape[1] != 660:
            spect.resize(128,660, refcheck=False)

        mel_spec.append(spect)

X_2 = np.array(mel_spec)
print(X_2.shape)        # (4994, 128, 660) 
print('FMA Finish!!')


X = np.concatenate((X_1, X_2))
X = X.reshape(X.shape[0], 128, 660, 1)
print(X.shape) # (6194, 128, 660, 1)

np.save('./project/mini/data/X.npy', arr=X)
print('save X!!')


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

y_label = pd.Series(labels).map(label_dict)
y_label.to_csv('./project/mini/data/y_label.csv', index=False)

y = pd.Series(labels).map(label_dict).values
y = to_categorical(y)
print(y.shape)  # (6194, 13)

np.save('./project/mini/data/y.npy', arr=y)
print('save y')
