import numpy as np
import librosa
from tensorflow.keras.models import load_model

import warnings
warnings.filterwarnings('ignore')

count = 0
model = load_model('./project/mini/data/genre_model_resnet50_0.6521.hdf5')
# model = load_model('./project/mini/data/genre_model_resnet34.hdf5')
# model = load_model('./project/mini/data/genre_model_resnet18.hdf5')

for i in ['dance', 'ballad', 'rock', 'hiphop', 'pop', 'blues', 'folk' ,'jazz', 'country', 'classical']:
    for j in range(10):
        test_music = './project/mini/data/{}.{}.mp3'.format(i,j)
        y, sr = librosa.load(test_music)

        mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
        mel_spect = librosa.power_to_db(mel_spect, ref=np.max)

        if mel_spect.shape[1] != 660:
            mel_spect.resize(128,660, refcheck=False)

        test_data = mel_spect.reshape(1, 128, 660, 1) / -80

        label_number = np.argmax(model.predict(test_data), axis=1)

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

        # print(label_dict.get(label_number[0]))

        print(test_music.split('/')[-1], '의 장르는', label_dict.get(label_number[0]), '입니다!!')
        print(np.round(model.predict(test_data)[0][label_number][0] * 100, 2) , "% 으로 예상됩니다.")
        print()
        if i == label_dict.get(label_number[0]):
            count += 1


print('100개 중', count, '개를 맞췄습니다!!')
print(count / 100 ,'% 정확도를 가집니다!!')