import numpy as np
import cv2
 
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

for i in range(np.array(alphabet).shape[0]):
    alphabet_ = resize_to_pixel(64, alphabet[i])
    cv2.imshow('img', alphabet_)
    cv2.waitKey(0)


'''==============================================================='''
# Import the modules
import cv2
import numpy as np
 
# Read the input image 
i = 0
image_path = '../data/dirty_mnist_2nd/{:05d}.png'.format(i)
im = cv2.imread(image_path)
 
# Convert to grayscale and apply Gaussian filtering
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gray = np.where(im_gray>=255, im_gray, 0)
im_gray = cv2.dilate(im_gray, kernel=np.ones((2,2), np.uint8), iterations=1)
im_gray = cv2.medianBlur(src=im_gray, ksize=5)

im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
 
# Threshold the image
ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)
 
# Find contours in the image
ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
# Get rectangles contains each contour
rects = [cv2.boundingRect(ctr) for ctr in ctrs]
 
# For each rectangular region
for rect in rects:
    # Draw the rectangles
    cv2.rectangle(im_gray, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 0, 0), 3) 
 
    # Make the rectangular region around the digit
    leng = int(rect[3] * 1.6)
    pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
    pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
    roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
    if roi.size <= 0:
        continue
 
    # Resize the image
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    roi = cv2.dilate(roi, (3, 3))
 
cv2.imshow("Resulting Image with Rectangular ROIs", im_gray)
cv2.waitKey()