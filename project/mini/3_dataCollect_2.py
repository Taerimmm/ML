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
