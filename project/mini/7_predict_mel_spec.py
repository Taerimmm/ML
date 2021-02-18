import numpy as np
import librosa
from tensorflow.keras.models import load_model

import warnings
warnings.filterwarnings('ignore')

test_music = '../data/project_data/mini/unclassified/091899.mp3'
y, sr = librosa.load(test_music)

mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
mel_spect = librosa.power_to_db(mel_spect, ref=np.max)

if mel_spect.shape[1] != 660:
    mel_spect.resize(128,660, refcheck=False)

test_data = mel_spect.reshape(1, 128, 660, 1) / -80
print(test_data.shape)

model = load_model('./project/mini/data/genre_model_resnet50.hdf5')

label_number = np.argmax(model.predict(test_data), axis=1)

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
