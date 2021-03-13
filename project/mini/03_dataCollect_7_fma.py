import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display

import warnings
warnings.filterwarnings('ignore')

''' FMA '''

# MFCC

for dir in os.scandir('../data/project_data/mini/fma'):
    print(str(dir)[-5:-2])

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
    mfccs_14 = []
    mfccs_15 = []
    mfccs_16 = []
    mfccs_17 = []
    mfccs_18 = []
    mfccs_19 = []
    mfccs_20 = []
    
    for file in librosa.util.find_files(dir):
        label = str(file).split('.')[0][43:]

        if label in ['experimental', 'instrumental', 'international']:
            continue    

        y, sr = librosa.core.load(file)
        
        filename = str(file)[43:]
        files.append(filename)

        label = str(file).split('.')[0][43:]
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
        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=512, n_mfcc=20)
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
        mfccs_14.append(mfcc_scaled[13])
        mfccs_15.append(mfcc_scaled[14])
        mfccs_16.append(mfcc_scaled[15])
        mfccs_17.append(mfcc_scaled[16])
        mfccs_18.append(mfcc_scaled[17])
        mfccs_19.append(mfcc_scaled[18])
        mfccs_20.append(mfcc_scaled[19])


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
            'mfcc_14': mfccs_14,
            'mfcc_15': mfccs_15,
            'mfcc_16': mfccs_16,
            'mfcc_17': mfccs_17,
            'mfcc_18': mfccs_18,
            'mfcc_19': mfccs_19,
            'mfcc_20': mfccs_20,
            'labels': labels
        })

    print(df)
    df.to_csv('./project/mini/data/fma_genres_mfcc_{}.csv'.format(str(dir)[-5:-2]), index=False)
