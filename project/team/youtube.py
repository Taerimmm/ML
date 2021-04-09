# 참고 : https://seolin.tistory.com/93

from pytube import YouTube

# 유튜브 전용 인스턴스 생성
par = 'https://www.youtube.com/watch?v=v7bnOxV4jAc'
yt = YouTube(par)

print(yt.title)

# 화질 확인
for e in yt.streams.filter(file_extension='mp4').all():
    print(str(e))

# 음성이 없는 영상 다운로드
# order_by('resolution').desc().first() 로 해상도 가장 좋은 영상 다운로드
yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first().download('./project/team/data', filename='LILAC')
print('success')
