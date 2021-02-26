import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.layers import BatchNormalization, ZeroPadding2D, Activation, Add, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

train_data = pd.read_csv('../data/mnist_data/train.csv', index_col=0, header=0)

print(train_data)

img = train_data.iloc[0,2:].values.reshape(28,28).astype(np.uint8)

img_2 = cv2.dilate(img, kernel=np.ones((2,2), np.uint8), iterations=1)
img_2 = cv2.medianBlur(src=img, ksize=5)

img_2 = np.where(img_2>=10, img_2, 0)

print(img_2.shape)
print(img_2)

'''
# 그림 확인
cv2.imshow('before',img)
cv2.imshow('after',img_2)
cv2.waitKey(0)
cv2.destroyAllWindows()

plt.imshow(img_2)
plt.show()
'''

datagen = ImageDataGenerator(
    rotation_range=360
)

train_letter = train_data['letter'].values

x_train = train_data.drop(['digit', 'letter'], axis=1).values
x_train = x_train.reshape(-1, 28, 28, 1)
x_train = x_train/255

print(x_train.shape)    # (2048, 28, 28, 1)

y = train_data['letter']

alpha_2_num = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7, 'I':8, 'J':9, 'K':10,
               'L':11, 'M':12, 'N':13, 'O':14, 'P':15, 'Q':16, 'R':17, 'S':18, 'T':19, 'U':20,
               'V':21, 'W':22, 'X':23, 'Y':24, 'Z':25}
y = y.map(alpha_2_num)

y_train = np.zeros((len(y), len(y.unique())))
for i, letter in enumerate(y):
    y_train[i, letter] = 1

print(y_train)
print(y_train.shape)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

# 모델
input_tensor = Input(shape=x_train.shape[1:], dtype='float32', name='input')

def conv1_layer(x):    
    x = ZeroPadding2D(padding=(3, 3))(x)
    x = Conv2D(64, (7, 7), strides=(1, 1))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1,1))(x)
 
    return x   

def conv2_layer(x):         
    x = MaxPooling2D((3, 3), 2)(x)     
 
    shortcut = x
 
    for i in range(2):
        if (i == 0):            
            x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
 
            x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
            shortcut = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(shortcut)            
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)
 
            x = Add()([x, shortcut])
            x = Activation('relu')(x)
            
            shortcut = x
 
        else:
            x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
 
            x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)            
 
            x = Add()([x, shortcut])   
            x = Activation('relu')(x)  
 
            shortcut = x        
    
    return x

def conv3_layer(x):        
    shortcut = x    
    
    for i in range(2):     
        if(i == 0): 
            x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)  
 
            x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
            shortcut = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)            
 
            x = Add()([x, shortcut])    
            x = Activation('relu')(x)    
 
            shortcut = x              
        
        else:
            x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
 
            x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)            
 
            x = Add()([x, shortcut])     
            x = Activation('relu')(x)
 
            shortcut = x      
            
    return x
 
def conv4_layer(x):
    shortcut = x        
  
    for i in range(2):     
        if(i == 0):
            x = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)  
 
            x = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
            shortcut = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)
 
            x = Add()([x, shortcut]) 
            x = Activation('relu')(x)
 
            shortcut = x               
        
        else:
            x = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
 
            x = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)            
 
            x = Add()([x, shortcut])    
            x = Activation('relu')(x)
 
            shortcut = x      
 
    return x
 
def conv5_layer(x):
    shortcut = x    
  
    for i in range(2):     
        if(i == 0):
            x = Conv2D(512, (3, 3), strides=(2, 2), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)  
 
            x = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
            shortcut = Conv2D(512, (3, 3), strides=(2, 2), padding='same')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)            
 
            x = Add()([x, shortcut])  
            x = Activation('relu')(x)      
 
            shortcut = x               
        
        else:
            x = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
 
            x = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)           
            
            x = Add()([x, shortcut]) 
            x = Activation('relu')(x)       
 
            shortcut = x                  
 
    return x
 
x = conv1_layer(input_tensor)
x = conv2_layer(x)
x = conv3_layer(x)
x = conv4_layer(x)
x = conv5_layer(x)
 
x = GlobalAveragePooling2D()(x)
output_tensor = Dense(26, activation='softmax')(x)

resnet18 = Model(input_tensor, output_tensor)

resnet18.summary()

model = resnet18
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
file_path = './dacon3/data/mnist_alpha_resnet_test.hdf5'
es = EarlyStopping(monitor='val_accuracy', patience=80)
cp = ModelCheckpoint(filepath=file_path, monitor='val_accuracy', save_best_only=True)
lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.8, patience=30)
# history = model.fit(x_train, y_train, epochs=5000, batch_size=32, validation_data=(x_val, y_val), verbose=2, callbacks=[es,cp,lr])

hist = model.fit_generator(datagen.flow(x_train, y_train, batch_size=16), epochs=2000,
            validation_data=(datagen.flow(x_val, y_val)), verbose=2, callbacks=[es, cp, lr])