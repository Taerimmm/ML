# 참고 : https://shilan.tistory.com/entry/Python%EC%9D%84-%EC%9D%B4%EC%9A%A9%ED%95%98%EC%97%AC-%EB%8F%99%EC%98%81%EC%83%81%EC%9C%BC%EB%A1%9C%EB%B6%80%ED%84%B0-%EC%9D%B4%EB%AF%B8%EC%A7%80%EC%B6%94%EC%B6%9C-Pythonv27-OpenCV-Windows

# __authour__ = 'TR'

import cv2

# 영상의 의미지를 연속적으로 캡쳐할 수 있게 하는 class
vidcap = cv2.VideoCapture('./project/team/data/LILAC.mp4')

count = 0

# Frame 수
length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
print( length )

# FPS
fps = int(vidcap.get(cv2.CAP_PROP_FPS))
print(fps)

# get() 함수를 이용하여 전체 프레임 중 1/20의 프레임만 가져와 저장
while (vidcap.isOpened()):
    ret, image = vidcap.read()
    
    if int(vidcap.get(1)) % (fps * 10) == 0:
        print('Save frame number :', str(int(vidcap.get(1))))
        cv2.imwrite('./project/team/data/{}.jpg'.format(count), image)
        print('Save frame %d.jpg' %count)
        count += 1
    
    elif vidcap.get(1) == int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)):
        break
