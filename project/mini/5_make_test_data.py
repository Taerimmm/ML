# test data 만들기
# 1. pytube를 이용해서 원하는 음원을 받은 후 30초로 잘라서 활용 or 그대로 활용 (비교해 볼 것)
#    ---> https://kaen2891.tistory.com/32 참고하면 30초로 자를 수 있을 듯...?

# 2. 3_dataCollect_2 에서 제외된 data 활용
#    ---> 번거로운 일이지만 오히려 편할 수 있다 / 대신 몇개 안됨

from pytube import YouTube
import glob
import os.path
import numpy as np
import librosa
import soundfile as sf

import warnings
warnings.filterwarnings('ignore')

# 유튜브 전용 인스턴스 생성
par = 'https://www.youtube.com/watch?v=noYRi8bi0aY&ab_channel=KangHero'
yt = YouTube(par)
yt.streams.filter(only_audio=True).all()

# 특정영상 다운로드
file_name = '9'
yt.streams.filter(only_audio=True).first().download('./project/mini/data', filename=file_name) # filename 수정해서 원하는 file명으로 고치기.
print('success')

# 확장자 변경
os.chdir('./project/mini/data')
files = glob.glob("*.mp4")
for x in files:
    if not os.path.isdir(x):
        filename = os.path.splitext(x)
        try:
            os.rename(x, filename[0] + '.mp3')
        except:
            pass
print('success')

# 30초로 자르기 
y, sr = librosa.load('{}.mp3'.format(file_name))

resize_time = sr * 30

# with sf.SoundFile('{}.mp3'.format('ballad.' + file_name), 'w', sr, channels=1, format='wav') as f:
#     f.write(y[:resize_time])
# print('Finish!!')
with sf.SoundFile('{}.mp3'.format('classical.' + file_name), 'w', sr, channels=1, format='wav') as f:
    f.write(y[resize_time*2:resize_time*3])
print('Finish!!')
