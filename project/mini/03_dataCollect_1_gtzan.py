'''
우선 해야할 것이
1. genres_original 에 있는 1000개 wav 파일 load 해서 data화
2. fma_small 에 있는 mp3 파일 genre 매칭해서 data화

이것들이 끝나고 나면 
pytube.py 파일을 이용해 새로운 genre mp3 파일 100개 수집
해당 mp3 파일을 30초로 자르는 방법이 있을까 체크하기
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display

import warnings
warnings.filterwarnings('ignore')

''' GTZAN '''

''' Function to Read and Extract Mel Spectrograms from Audio Files '''
# # Creating an empty list to store sizes in
# sizes = []

# Looping through each audio file
for dir in os.scandir('../data/project_data/mini/genre'):
    sizes=[]
    for file in os.scandir(dir):
        # Loading in the audio file
        y, sr = librosa.core.load(file)

        # Computing the mel spectrograms
        spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
        spect = librosa.power_to_db(spect, ref=np.max)

        # Adding the size to the list
        sizes.append(spect.shape)
    
    # Checking if all sizes are the same
    print(f'The sizes of all the mel spectrograms in our data set are equal: {len(set(sizes)) == 1}')

    # Checking the max size
    print(f'The maximum size is: {max(sizes)}')



''' Function to Read and Extract Numeric Features from Audio Files '''
# MFCC
files = []
labels = []
zcrs = []
spec_centroids = []
spec_rolloffs = []
mfccs_1 = []
mfccs_2 = []
mfccs_3 = []
mfccs_4 = []
mfccs_5 = []
mfccs_6 = []
mfccs_7 = []
mfccs_8 = []
mfccs_9 = []
mfccs_10 = []
mfccs_11 = []
mfccs_12 = []
mfccs_13 = []
for dir in os.scandir('../data/project_data/mini/genre'):
    for file in os.scandir(dir):
        y, sr = librosa.core.load(file)

        filename = str(file).split()[1][1:-2]
        files.append(filename)

        label = str(file).split('.')[0][11:]
        labels.append(label)

        # Calculating zero-crossing rates
        zcr = librosa.feature.zero_crossing_rate(y)
        zcrs.append(np.mean(zcr))

        # Calculating the spectral centroids
        spec_centroid = librosa.feature.spectral_centroid(y)
        spec_centroids.append(np.mean(spec_centroid))

        # Calculating the spectral rolloffs
        spec_rolloff = librosa.feature.spectral_rolloff(y)
        spec_rolloffs.append(np.mean(spec_rolloff))

        # Calculating the first 13 mfcc coefficients
        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=512, n_mfcc=13)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        mfccs_1.append(mfcc_scaled[0])
        mfccs_2.append(mfcc_scaled[1])
        mfccs_3.append(mfcc_scaled[2])
        mfccs_4.append(mfcc_scaled[3])
        mfccs_5.append(mfcc_scaled[4])
        mfccs_6.append(mfcc_scaled[5])
        mfccs_7.append(mfcc_scaled[6])
        mfccs_8.append(mfcc_scaled[7])
        mfccs_9.append(mfcc_scaled[8])
        mfccs_10.append(mfcc_scaled[9])
        mfccs_11.append(mfcc_scaled[10])
        mfccs_12.append(mfcc_scaled[11])
        mfccs_13.append(mfcc_scaled[12])


df = pd.DataFrame({
        'files': files,
        'zero_crossing_rate': zcrs,
        'spectral_centroid': spec_centroids,
        'spectral_rolloff': spec_rolloffs,
        'mfcc_1': mfccs_1,
        'mfcc_2': mfccs_2,
        'mfcc_3': mfccs_3,
        'mfcc_4': mfccs_4,
        'mfcc_5': mfccs_5,
        'mfcc_6': mfccs_6,
        'mfcc_7': mfccs_7,
        'mfcc_8': mfccs_8,
        'mfcc_9': mfccs_9,
        'mfcc_10': mfccs_10,
        'mfcc_11': mfccs_11,
        'mfcc_12': mfccs_12,
        'mfcc_13': mfccs_13,
        'labels': labels
    })

print(df)
df.to_csv('./project/mini/data/gtzan_genres_mfcc.csv', index=False)
