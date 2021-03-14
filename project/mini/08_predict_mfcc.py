import numpy as np
import pandas as pd
import librosa
from tensorflow.keras.models import load_model

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings('ignore')

test_music = './project/mini/data/rock.3.mp3'

y, sr = librosa.core.load(test_music)

features = []

# Calculating zero-crossing rates
zcr = librosa.feature.zero_crossing_rate(y)
features.append(np.mean(zcr))

# Calculating the spectral centroids
spec_centroid = librosa.feature.spectral_centroid(y)
features.append(np.mean(spec_centroid))

# Calculating the spectral rolloffs
spec_rolloff = librosa.feature.spectral_rolloff(y)
features.append(np.mean(spec_rolloff))

# Calculating the first 13 mfcc coefficients
mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=512, n_mfcc=20)
mfcc_scaled = np.mean(mfcc.T, axis=0)
for i in mfcc_scaled:
    features.append(i)


data = pd.read_csv('./project/mini/data/total_genres_mfcc.csv', header=0)
x = data.iloc[:,1:-1]
y = data.iloc[:,-1]
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
scaler = MinMaxScaler()
x_train = scaler.fit(x_train)

features = scaler.transform(np.array(features).reshape(1,-1))


model = load_model('./project/mini/data/genre_mfcc_model.hdf5')

label_number = np.argmax(model.predict(features), axis=1)

print(label_number[0])

label_dict = {
    0:'hiphop',
    1:'rock',
    2:'pop',
    3:'folk',
    4:'electronic',
    5:'jazz',
    6:'blues',
    7:'classical',
    8:'reggae',
    9:'disco',
    10:'country',
    11:'ballad',
    12:'dance'
}

print(label_dict.get(label_number[0]))
print(test_music.split('/')[-1], '의 장르는', label_dict.get(label_number[0]), '입니다!!')
