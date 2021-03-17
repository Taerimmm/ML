import os
import numpy as np
import pandas as pd
import cv2

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# img = cv2.imread('../data/LPD_competition/train/0/0.jpg')
# cv2.imshow('img', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


labels = os.listdir('../data/LPD_competition/train')
print(labels)

for dir in os.scandir('../data/LPD_competition/train'):
    print(dir)
    for file in os.scandir(dir):
        print(file)
    break


# Found 39000 images belonging to 1000 classes.
train_generator = ImageDataGenerator(rescale=1./255, validation_split=0.2).flow_from_directory(
    '../data/LPD_competition/train',
    target_size=(128,128),
    # color_mode='grayscale',
    subset='training'
)
# Found 9000 images belonging to 1000 classes.
val_generator = ImageDataGenerator(rescale=1./255, validation_split=0.2).flow_from_directory(
    '../data/LPD_competition/train',
    target_size=(128,128),
    # color_mode='grayscale',
    subset='validation'
)

print(train_generator)
print(val_generator)

# Found 72000 images belonging to 1 classes.
test_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
    '../data/LPD_competition',
    target_size=(128,128),
    # color_mode='grayscale',
    classes=['test'],
    shuffle=False,
    class_mode=None
)

print(test_generator)

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(128,128,3))
vgg16.trainable = False

input_tensor = Input(shape=(128,128,3))
layer = vgg16(input_tensor)
layer = Flatten()(layer)
layer = Dense(4096)(layer)
layer = Dense(4096)(layer)
layer = Dense(1000, activation='softmax')(layer)

model = Model(inputs=input_tensor, outputs=layer)

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
path = './Lotte/model.hdf5'
es = EarlyStopping(monitor='val_accuracy', patience=30)
cp = ModelCheckpoint(path, monitor='val_accuracy', save_best_only=True)
lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.8, patience=10)
model.fit(train_generator, epochs=2000, batch_size=32, validation_data=val_generator, callbacks=[es,cp,lr])

pred = model.predict(test_generator)
print(np.argmax(pred,1))

answer = pd.read_csv('./Lotte/sample.csv', header=0)

answer.iloc[:,1] = np.argmax(pred,1)
print(answer)
answer.to_csv('./Lotte/submission.csv', index=False)

'''
#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡparameterㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#
path ='C:/Users/ai/Desktop/miniproject/myData' # 경로설정 
labelFile = 'C:/Users/ai/Desktop/miniproject/labels.csv' # label이 들어있는 파일 
batch_size_val=50  
steps_per_epoch_val=2000
epochs_val=10
imageDimesions = (32,32,3)
testRatio = 0.2    # test 비율, 1000개면 test가 200개 
validationRatio = 0.2 # val 비율, 800개면 val이 160개
#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#

#ㅡㅡㅡㅡㅡㅡㅡㅡㅡ이미지 수집ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#
count = 0
images = []
classNo = []
myList = os.listdir(path) #위에 경로에 모든 파일 
print("Total Classes Detected:",len(myList)) #파일갯수 = 43
noOfClasses=len(myList)
print("Importing Classes.....") # 클래스 수집하는중

for x in range (0,len(myList)): #0에서 파일갯수 만큼 42번 까지 반복 (0개부터 시작이어서 42번에 끝나면 총 43개)
    myPicList = os.listdir(path+"/"+str(count)) # count0 부터 수집하기 시작
    for y in myPicList:
        curImg = cv2.imread(path+"/"+str(count)+"/"+y) # 수집되는 폴더마다 안에 이미지를 cv2로 수집 
        images.append(curImg) # 이미지는 curImg에 넣음
        classNo.append(count) # count증가로 classNo에 넣음
    print(count, end =" ")
    count +=1
print(" ") # 안써주면 42 이후로 붙어서 나옴.

images = np.array(images)

# print(classNo)

classNo = np.array(classNo)


############################### Split Data 

X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio) #test 비율 = 0.2 

X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validationRatio) #val 비율 = 0.2

# X_train = 이미지 훈련 배열 

# y_train = y 값 도출 

# print(X_train.shape) #(22265, 32, 32 ,3)
# print(X_validation.shape) #(5567, 32, 32, 3)

# print(y_train.shape) #(22265,)
# print(y_validation.shape) #(5567,)

############################### TO CHECK IF NUMBER OF IMAGES MATCHES TO NUMBER OF LABELS FOR EACH DATA SET

# print(classNo)

print(classNo.shape)

print("Data Shapes")

print("Train",end = "");print(X_train.shape,y_train.shape)

print("Validation",end = "");print(X_validation.shape,y_validation.shape)

print("Test",end = "");print(X_test.shape,y_test.shape)

assert(X_train.shape[0]==y_train.shape[0]), "22271, 22271 로 같다 틀리다면 이문장과 함께 error가 뜬다/ assert <표현식> [, '진단 메시지']"

assert(X_validation.shape[0]==y_validation.shape[0])

assert(X_test.shape[0]==y_test.shape[0])

assert(X_train.shape[1:]==(imageDimesions))

assert(X_validation.shape[1:]==(imageDimesions))

assert(X_test.shape[1:]==(imageDimesions))

#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ csv 파일 읽어들인다 ㅡㅡㅡㅡㅡㅡㅡㅡㅡ#  

# csv 파일 안에는 클래스 0~42 각각의 숫자마다 label이 붙어있다.

data=pd.read_csv(labelFile)

 

print("data shape ", data.shape ,type(data))


#ㅡㅡㅡㅡㅡㅡㅡㅡㅡ class별 샘플이미지 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#

num_of_samples = []

cols = 5 #데이터 파일안에 이미지를 5개를 보여줌 

num_classes = noOfClasses  # 데이터 파일 갯수 0~42 총 43개 

fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5, 300)) 

fig.tight_layout() 

for i in range(cols):

    for j,row in data.iterrows():

        x_selected = X_train[y_train == j]

        axs[j][i].imshow(x_selected[random.randint(0, len(x_selected)- 1), :, :], cmap=plt.get_cmap("gray")) #cmap- 차트종류 colormap

        axs[j][i].axis("off")

        if i == 2:

            axs[j][i].set_title(str(j)+ "-"+row["Name"])

            num_of_samples.append(len(x_selected))

 

 

#ㅡㅡㅡㅡㅡㅡㅡ각 범주에대해 표본이 없는걸 그래프 바로 표시ㅡㅡㅡㅡㅡ#

 

 

# plt.figure(figsize=(12, 4))

# plt.bar(range(0, num_classes), num_of_samples)

# plt.title("Distribution of the training dataset")

# plt.xlabel("Class number")

# plt.ylabel("Number of images")

# plt.show()

 

############################### PREPROCESSING THE IMAGES

 

def grayscale(img): # 회색으로 변환 

    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    return img

def equalize(img):  # 이미지 평준화

    img =cv2.equalizeHist(img) 

    return img

def preprocessing(img): # 위에꺼와 이미지 전처리 , 0~1사이값으로 표준화 

    img = grayscale(img)      

    img = equalize(img)      

    img = img/255           

    return img

 

X_train=np.array(list(map(preprocessing,X_train)))  # 모든 이미지를 위에있는 전처리로 사용하고 리스트로 변환 map은 for문축소

X_validation=np.array(list(map(preprocessing,X_validation)))

X_test=np.array(list(map(preprocessing,X_test)))

# cv2.imshow("GrayScale Images",X_train[random.randint(0,len(X_train)-1)]) # 교육이 제대로 수행되었는지.

# cv2.waitKey(0) # 0이면 무한대기 

# cv2.destroyAllWindows()

# random.randint(1,20) # 1부터 19까지 랜덤숫자 1개 

 

############################### ADD A DEPTH OF 1

X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)

X_validation=X_validation.reshape(X_validation.shape[0],X_validation.shape[1],X_validation.shape[2],1)

X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)

 

 

#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ이미지 증강ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#

dataGen= ImageDataGenerator(width_shift_range=0.1,   # 0.1 = 10%     IF MORE THAN 1 E.G 10 THEN IT REFFERS TO NO. OF  PIXELS EG 10 PIXELS

                            height_shift_range=0.1,

                            zoom_range=0.2,  # 0.2 MEANS CAN GO FROM 0.8 TO 1.2

                            shear_range=0.1,  # MAGNITUDE OF SHEAR ANGLE

                            rotation_range=10)  # DEGREES

dataGen.fit(X_train)

batches= dataGen.flow(X_train,y_train,batch_size=20)  # 호출할 때마다 생성되는 이미지 수

X_batch,y_batch = next(batches)

 

#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ 분할된 이미지 샘플을 표시ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#

# fig,axs=plt.subplots(1,15,figsize=(20,5))

# fig.tight_layout()

 

# for i in range(15):

#     axs[i].imshow(X_batch[i].reshape(imageDimesions[0],imageDimesions[1]))

#     axs[i].axis('off')

# plt.show()

 

 

y_train = to_categorical(y_train,noOfClasses) #noOfClasses - 파일 갯수 43개

y_validation = to_categorical(y_validation,noOfClasses)

y_test = to_categorical(y_test,noOfClasses)

 

############################### CONVOLUTION NEURAL NETWORK MODEL

def myModel():

    no_Of_Filters=60

    size_of_Filter=(5,5) # THIS IS THE KERNEL THAT MOVE AROUND THE IMAGE TO GET THE FEATURES.

                         # THIS WOULD REMOVE 2 PIXELS FROM EACH BORDER WHEN USING 32 32 IMAGE

    size_of_Filter2=(3,3)

    size_of_pool=(2,2)  # SCALE DOWN ALL FEATURE MAP TO GERNALIZE MORE, TO REDUCE OVERFITTING

    no_Of_Nodes = 500   # NO. OF NODES IN HIDDEN LAYERS

    model= Sequential()

    model.add((Conv2D(no_Of_Filters,size_of_Filter,input_shape=(imageDimesions[0],imageDimesions[1],1),activation='relu')))  # ADDING MORE CONVOLUTION LAYERS = LESS FEATURES BUT CAN CAUSE ACCURACY TO INCREASE

    model.add((Conv2D(no_Of_Filters, size_of_Filter, activation='relu')))

    model.add(MaxPooling2D(pool_size=size_of_pool)) # DOES NOT EFFECT THE DEPTH/NO OF FILTERS

 

    model.add((Conv2D(no_Of_Filters//2, size_of_Filter2,activation='relu')))

    model.add((Conv2D(no_Of_Filters // 2, size_of_Filter2, activation='relu')))

    model.add(MaxPooling2D(pool_size=size_of_pool))

    model.add(Dropout(0.5))

 

    model.add(Flatten())

    model.add(Dense(no_Of_Nodes,activation='relu'))

    model.add(Dropout(0.5)) # INPUTS NODES TO DROP WITH EACH UPDATE 1 ALL 0 NONE

    model.add(Dense(noOfClasses,activation='softmax')) # OUTPUT LAYER

    # COMPILE MODEL

    model.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])

    return model

 

 

############################### TRAIN

model = myModel()

# print(model.summary())

history=model.fit_generator(dataGen.flow(X_train,y_train,batch_size=20),epochs=100,validation_data=(X_validation,y_validation),shuffle=1)

# history=model.fit_generator(dataGen.flow(X_train,y_train,batch_size=batch_size_val),steps_per_epoch=steps_per_epoch_val,epochs=epochs_val,validation_data=(X_validation,y_validation),shuffle=1)

 

############################### PLOT

# plt.figure(1)

# plt.plot(history.history['loss'])

# plt.plot(history.history['val_loss'])

# plt.legend(['training','validation'])

# plt.title('loss')

# plt.xlabel('epoch')

# plt.figure(2)

# plt.plot(history.history['accuracy'])

# plt.plot(history.history['val_accuracy'])

# plt.legend(['training','validation'])

# plt.title('Acurracy')

# plt.xlabel('epoch')

# plt.show()

# score =model.evaluate(X_test,y_test,verbose=0)

# print('Test Score:',score[0])

# print('Test Accuracy:',score[1])

 

 

#ㅡㅡㅡㅡㅡㅡㅡㅡㅡpickle 안됨 ㅡㅡㅡㅡㅡㅡㅡㅡㅡ#

# pickle_out= open("./model_trained.p","wb")  # wb = WRITE BYTE

# pickle.dump(model, pickle_out)

# pickle_out.close()

# cv2.waitKey(0)

# model.save("my_model")

# model.save_weights("weights.h5")

# generate 와 pickle, model 저장 은 안되는거 같다 

 

import numpy as np

import cv2

import pickle

 

#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#

 

frameWidth= 640         # CAMERA RESOLUTION

frameHeight = 480

brightness = 180

threshold = 0.75         # PROBABLITY THRESHOLD

font = cv2.FONT_HERSHEY_SIMPLEX

#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#

 

# image2 = cv2.imread('C:/Users/ai/Desktop/sample.png') 

# image2 = preprocessing(image2)

# cv2.imshow('result', image2)

# cv2.waitKey(0) 

 

# SETUP THE VIDEO CAMERA

cap = cv2.VideoCapture(0)

cap.set(3, frameWidth)

cap.set(4, frameHeight)

cap.set(10, brightness)

# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ IMPORT THE TRANNIED MODEL ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#

# pickle_in=open("model_trained.p","rb")  ## rb = READ BYTE

# model=pickle.load(pickle_in)

 

 

def getCalssName(classNo):

    if   classNo == 0: return 'Speed Limit 20 km/h'

    elif classNo == 1: return 'Speed Limit 30 km/h'

    elif classNo == 2: return 'Speed Limit 50 km/h'

    elif classNo == 3: return 'Speed Limit 60 km/h'

    elif classNo == 4: return 'Speed Limit 70 km/h'

    elif classNo == 5: return 'Speed Limit 80 km/h'

    elif classNo == 6: return 'End of Speed Limit 80 km/h'

    elif classNo == 7: return 'Speed Limit 100 km/h'

    elif classNo == 8: return 'Speed Limit 120 km/h'

    elif classNo == 9: return 'No passing'

    elif classNo == 10: return 'No passing for vechiles over 3.5 metric tons'

    elif classNo == 11: return 'Right-of-way at the next intersection'

    elif classNo == 12: return 'Priority road'

    elif classNo == 13: return 'Yield'

    elif classNo == 14: return 'Stop'

    elif classNo == 15: return 'No vechiles'

    elif classNo == 16: return 'Vechiles over 3.5 metric tons prohibited'

    elif classNo == 17: return 'No entry'

    elif classNo == 18: return 'General caution'

    elif classNo == 19: return 'Dangerous curve to the left'

    elif classNo == 20: return 'Dangerous curve to the right'

    elif classNo == 21: return 'Double curve'

    elif classNo == 22: return 'Bumpy road'

    elif classNo == 23: return 'Slippery road'

    elif classNo == 24: return 'Road narrows on the right'

    elif classNo == 25: return 'Road work'

    elif classNo == 26: return 'Traffic signals'

    elif classNo == 27: return 'Pedestrians'

    elif classNo == 28: return 'Children crossing'

    elif classNo == 29: return 'Bicycles crossing'

    elif classNo == 30: return 'Beware of ice/snow'

    elif classNo == 31: return 'Wild animals crossing'

    elif classNo == 32: return 'End of all speed and passing limits'

    elif classNo == 33: return 'Turn right ahead'

    elif classNo == 34: return 'Turn left ahead'

    elif classNo == 35: return 'Ahead only'

    elif classNo == 36: return 'Go straight or right'

    elif classNo == 37: return 'Go straight or left'

    elif classNo == 38: return 'Keep right'

    elif classNo == 39: return 'Keep left'

    elif classNo == 40: return 'Roundabout mandatory'

    elif classNo == 41: return 'End of no passing'

    elif classNo == 42: return 'End of no passing by vechiles over 3.5 metric tons'

 

model.save_weights("./data/save_miniproject_practice6.h5")

video_file = 'C:/Users/ai/Desktop/sample10.mp4' 

cap = cv2.VideoCapture(video_file)

if cap.isOpened():

    while True:

        ret, imgOrignal = cap.read()

        img = np.asarray(imgOrignal)

        img = cv2.resize(img, (32, 32))

        img = preprocessing(img)

 

        img = img.reshape(1, 32, 32, 1)

        cv2.putText(imgOrignal, "CLASS: " , (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.putText(imgOrignal, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

       

        predictions = model.predict(img)

        classIndex = model.predict_classes(img)

        probabilityValue =np.amax(predictions)

        if probabilityValue > threshold:

    

            cv2.putText(imgOrignal,str(classIndex)+" "+str(getCalssName(classIndex)), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.putText(imgOrignal, str(round(probabilityValue*100,2) )+"%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow("Result", imgOrignal)

        if ret:

            cv2.waitKey(100)

        else:

            break

else:

    print('cannot open the file')

 

cap.release()

cv2.destroyAllWindows()
'''
 

 

'''

    # READ IMAGE

    success, imgOrignal = cap.read()

    

    # PROCESS IMAGE

    img = np.asarray(imgOrignal)

    img = cv2.resize(img, (32, 32))

    img = preprocessing(img)

    cv2.imshow("Processed Image", img)

    img = img.reshape(1, 32, 32, 1)

    cv2.putText(imgOrignal, "CLASS: " , (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.putText(imgOrignal, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

    # PREDICT IMAGE

    predictions = model.predict(img)

    classIndex = model.predict_classes(img)

    probabilityValue =np.amax(predictions)

    if probabilityValue > threshold:

    #print(getCalssName(classIndex))

        cv2.putText(imgOrignal,str(classIndex)+" "+str(getCalssName(classIndex)), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.putText(imgOrignal, str(round(probabilityValue*100,2) )+"%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("Result", imgOrignal)

    

    if cv2.waitKey(1) and 0xFF == ord('q'):

        break

    '''