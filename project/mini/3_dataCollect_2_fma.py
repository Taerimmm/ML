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

''' FMA '''

# Change filename!!
# GTZAN 과 동일한 형태로 만들기

genre = pd.read_csv('../data/project_data/mini/2_fma_metadata/fma_genre.csv', index_col=0, header=0, engine='python')
print(genre.head())

for dir in os.scandir('../data/project_data/mini/fma'):
    for file in librosa.util.find_files(dir):
        track_id = os.path.split(file)[1].split(sep='.')[0].lstrip('0')
        # print(track_id)

        try:
            file_genre = genre.loc[int(track_id),:][0]
        except:
            pass
        # print(file_genre)

        print(os.path.split(file))
        # print(os.path.split(file)[0] + '\\' + file_genre.lower() + '.' + os.path.split(file)[1])
        try:
            os.rename(file, os.path.split(file)[0] + '\\' + file_genre.lower() + '.' + os.path.split(file)[1])
        except:
            pass

print('Finish!!')

# 장르 구별 안되있는 음원 제거 8000 --> 7973