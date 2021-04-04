import os
import cv2

for file in os.scandir('./project/team/filtered_img'):
    print(file)

    path = os.path.abspath(file)
    # print(path[35:])
    
    jpg = cv2.imread(path)
    jpg = cv2.resize(jpg, (1280,720))

    # cv2.imshow('jpg', jpg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # print('./project/team/resize_img/'+path[35:])    
    cv2.imwrite('./project/team/resize_img/' + path[35:], jpg)
