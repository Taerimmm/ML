import cv2
import os
import numpy as np
import pandas as pd
from PIL import Image

for file in os.scandir('./project/team/filtered_img'):
    print(file)

    path = os.path.abspath(file)
    # print(path[41:])
    
    jpg = cv2.imread(path)
    jpg = cv2.resize(jpg, (1280,720))

    # cv2.imshow('jpg', jpg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # print('./project/team/resize_img/'+path[41:])    
    cv2.imwrite('./project/team/resize_img/' + path[41:], jpg)



# jpg_1 = cv2.imread('481_pre.jpg')
# jpg_1 = cv2.resize(jpg_1, (1280,720))
# jpg = cv2.imread('481.jpg')
# print(jpg.shape)
# re_jpg = cv2.resize(jpg, (1280,720))
# print(re_jpg.shape)


# cv2.imshow('jpg_1', jpg_1)
# cv2.imshow('re_jpg', re_jpg)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imwrite()