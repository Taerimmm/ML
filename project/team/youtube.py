# 참고 : https://seolin.tistory.com/93

import os
import glob
from pytube import YouTube

# 유튜브 전용 인스턴스 생성
par = 'https://www.youtube.com/watch?v=TWj-8_-XnaU'
yt = YouTube(par)

print(yt.title)

# 화질 확인
for e in yt.streams.filter(file_extension='mp4').all():
    print(str(e))

# 음성이 없는 영상 다운로드
# order_by('resolution').desc().first() 로 해상도 가장 좋은 영상 다운로드
yt.streams.filter(progressive=False, file_extension='mp4').order_by('resolution').desc().first().download('./project/team/data/original_MV', filename='ONF')
print('success')

# 음성만 있는 영상 다운로드
yt.streams.filter(only_audio=True).first().download('./project/team/data/original_Music', filename='ONF') # filename 수정해서 원하는 file명으로 고치기.
print('success')

# 확장자 변경
os.chdir('./project/team/data/original_Music')
files = glob.glob("*.mp4")
for x in files:
    if not os.path.isdir(x):
        filename = os.path.splitext(x)
        try:
            os.rename(x, filename[0] + '.mp3')
        except:
            pass