import os
import numpy as np
import librosa
import librosa.display

import warnings
warnings.filterwarnings('ignore')

dir_max_size = []

# Looping through each audio file - GTZAN
for dir in os.scandir('../data/project_data/mini/new_genre'):
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
    
    dir_max_size.append(max(sizes))
print('GTZAN Finish!!')

print(dir_max_size)
print(max(dir_max_size))

# [(128, 646), (128, 647), (128, 657), (128, 654), (128, 646), (128, 653), (128, 660), (128, 657), (128, 647), (128, 647), (128, 655)]
# (128, 660)

# Looping through each audio file - FMA
for dir in os.scandir('../data/project_data/mini/fma'):
    sizes=[]
    for file in librosa.util.find_files(dir):
        # Loading in the audio file
        y, sr = librosa.load(file)

        # Computing the mel spectrograms
        spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
        spect = librosa.power_to_db(spect, ref=np.max)

        # Adding the size to the list
        sizes.append(spect.shape)
    
    # Checking if all sizes are the same
    print(f'The sizes of all the mel spectrograms in our data set are equal: {len(set(sizes)) == 1}')

    # Checking the max size
    print(f'The maximum size is: {max(sizes)}')

    dir_max_size.append(max(sizes))
print('FMA Finish!!')

print(dir_max_size)
print(max(dir_max_size))

# [(128, 647), (128, 657), (128, 654), (128, 653), (128, 660), (128, 657), (128, 647), (128, 647), (128, 647), (128, 655),
#  (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647),
#  (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647),
#  (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), 
#  (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647),
#  (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647),
#  (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647),
#  (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647),
#  (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647),
#  (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647),
#  (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647),
#  (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647),
#  (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647),
#  (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647),
#  (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647),
#  (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 647),
#  (128, 647), (128, 647), (128, 647), (128, 647), (128, 647), (128, 646)]
# (128, 660)
