# 참고 : https://theailearner.com/2018/10/15/creating-video-from-images-using-opencv-python/

import cv2
import numpy as np
import glob
 
img_array = []
for filename in sorted(glob.glob('C:/Study/project/team/data/frame/*.jpg'), key=lambda name:int(''.join(filter(str.isdigit, name)))):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
 
print(np.array(img_array).shape)

# 비디오 파일 이름 , fourcc , 초당 프레임 수, 프레임 사이즈
# cv2.VideoWriter_fourcc(*'DIVX') : DIVX MPEG-4 코덱
out = cv2.VideoWriter('./project/team/data/preview.avi', cv2.VideoWriter_fourcc(*'DIVX'), 23.98, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
