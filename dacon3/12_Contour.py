import numpy as np
import matplotlib.pyplot as plt
import cv2

from tensorflow.keras.models import load_model

def x_cord_contour(contour):
    # This function take a contour from findContours
    # it then outputs the x centroid coordinates
    
    
    M = cv2.moments(contour)
    return (int(M['m10']/M['m00']))
 
 
def makeSquare(not_square):
    # This function takes an image and makes the dimenions square
    # It adds black pixels as the padding where needed
    
    BLACK = [0,0,0]
    img_dim = not_square.shape
    height = img_dim[0]
    width = img_dim[1]
    #print("Height = ", height, "Width = ", width)
    if (height == width):
        square = not_square
        return square
    else:
        doublesize = cv2.resize(not_square,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)
        height = height * 2
        width = width * 2
        #print("New Height = ", height, "New Width = ", width)
        if (height > width):
            pad = int((height - width)/2)
            #print("Padding = ", pad)
            doublesize_square = cv2.copyMakeBorder(doublesize,0,0,pad,pad,cv2.BORDER_CONSTANT,value=[255,0,0])
        else:
            pad = int((width - height)/2)
            #print("Padding = ", pad)
            doublesize_square = cv2.copyMakeBorder(doublesize,pad,pad,0,0,cv2.BORDER_CONSTANT,value=[255,0,0])
    doublesize_square_dim = doublesize_square.shape
    #print("Sq Height = ", doublesize_square_dim[0], "Sq Width = ", doublesize_square_dim[1])
    return doublesize_square
 
 
def resize_to_pixel(dimensions, image):
    # This function then re-sizes an image to the specificied dimenions
    
    buffer_pix = 4
    dimensions  = dimensions - buffer_pix
    squared = image
    r = float(dimensions) / squared.shape[1]
    dim = (dimensions, int(squared.shape[0] * r))
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    img_dim2 = resized.shape
    height_r = img_dim2[0]
    width_r = img_dim2[1]
    BLACK = [0,0,0]
    if (height_r > width_r):
        resized = cv2.copyMakeBorder(resized,0,0,0,1,cv2.BORDER_CONSTANT,value=[255,0,0])
    if (height_r < width_r):
        resized = cv2.copyMakeBorder(resized,1,0,0,0,cv2.BORDER_CONSTANT,value=[255,0,0])
    p = 2
    ReSizedImg = cv2.copyMakeBorder(resized,p,p,p,p,cv2.BORDER_CONSTANT,value=[255,0,0])
    img_dim = ReSizedImg.shape
    height = img_dim[0]
    width = img_dim[1]
    #print("Padded Height = ", height, "Width = ", width)
    return ReSizedImg
    
i = 0
image_path = '../data/dirty_mnist_2nd/{:05d}.png'.format(i)
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = np.where(img>=255, img, 0)
img = cv2.dilate(img, kernel=np.ones((2,2), np.uint8), iterations=1)
img = cv2.medianBlur(src=img, ksize=5)
 
# Blur image then find edges using Canny 
blurred = cv2.GaussianBlur(img, (5, 5), 0)
cv2.imshow("blurred", blurred)
cv2.waitKey(0)
 
edged = cv2.Canny(blurred, 30, 150)
cv2.imshow("edged", edged)
cv2.waitKey(0)
 
# Fint Contours
contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
#Sort out contours left to right by using their x cordinates
filtered_contours = [c for c in contours if cv2.contourArea(c) > 10]
contours = sorted(filtered_contours, key = x_cord_contour, reverse = False)
# Create empty array to store entire number
full_number = []
 
# loop over the contours
alphabet = []
for c in contours:
    # compute the bounding box for the rectangle
    (x, y, w, h) = cv2.boundingRect(c)    
    
    #cv2.drawContours(image, contours, -1, (0,255,0), 3)
    #cv2.imshow("Contours", image)
 
    if w >= 5 and h >= 25:
        roi = blurred[y:y + h, x:x + w]
        ret, roi = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY_INV)
        squared = makeSquare(roi)
        final = resize_to_pixel(20, squared)
        alphabet.append(final)
        cv2.imshow("final", final)
        final_array = final.reshape((1,400))
        final_array = final_array.astype(np.float32)
        #ret, result, neighbours, dist = knn.findNearest(final_array, k=1)
        #number = str(int(float(result[0])))
        #full_number.append(number)
        # draw a rectangle around the digit, the show what the
        # digit was classified as
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        #cv2.putText(image, number, (x , y + 155),
        #    cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 2)
        cv2.imshow("image", img)
        cv2.waitKey(0) 

'''
for i in range(np.array(alphabet).shape[0]):
    alphabet_ = resize_to_pixel(64, alphabet[i])
    cv2.imshow('img', alphabet_)
    cv2.waitKey(0)
'''


model = load_model('./dacon3/data/mnist_alpha_resnet_test.hdf5')

num_2_alpha = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I', 9:'J', 10:'K',
               11:'L', 12:'M', 13:'N', 14:'O', 15:'P', 16:'Q', 17:'R', 18:'S', 19:'T', 20:'U',
               21:'V', 22:'W', 23:'X', 24:'Y', 25:'Z'}


for i in alphabet:
    plt.imshow(resize_to_pixel(28, i))
    plt.show()

    resized = resize_to_pixel(28, i).reshape(1, 28, 28, 1)
    alpha_pred =  model.predict(resized/255)
    print(alpha_pred)
    alpha_pred

    label_number = np.argmax(alpha_pred, axis=1)

    print(label_number[0])

    print(num_2_alpha.get(label_number[0]))

    # print(test_music.split('/')[-1], '의 장르는', label_dict.get(label_number[0]), '입니다!!')
    # print(np.round(model.predict(test_data)[0][label_number][0] * 100, 2) , "% 으로 예상됩니다.")

