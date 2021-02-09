import time
import datetime
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

start_time = time.time()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2)
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2)
test_datagen = ImageDataGenerator(
    rescale=1./255)

''' ImgDatagen으로 50000개 image 로드 해보기 '''
# Found 40000 images belonging to 1 classes.
train_generator = train_datagen.flow_from_directory(
    '../data', 
    classes=['dirty_mnist_2nd'],
    batch_size=40000, # batch_size 수정해서 fit or fit_generator에 사용할 것.
    target_size=(256, 256), 
    color_mode='grayscale',
    class_mode=None,
    subset='training')
# Found 10000 images belonging to 1 classes.
val_generator = val_datagen.flow_from_directory(
    '../data',
    classes=['dirty_mnist_2nd'],
    batch_size=10000, 
    target_size=(256, 256), 
    color_mode='grayscale',
    class_mode=None,
    subset='validation')

for i in train_generator:
    print(i)
    print(type(i))
    print(i.shape)
    break
    
end_time = time.time()
print('실행시간 :', str(datetime.timedelta(seconds=end_time-start_time)))

y_train = pd.read_csv('./dacon3/data/dirty_mnist_2nd_answer.csv', index_col=0, header=0)
print(y_train.shape)


'''
# generator 랑 y를 맞추자 !!

- validation_split을 하지 않고 shuffule=False로 하면 기존에 x_train set과 일치 하는가? (3번 참고)
  만약에 일치하면 kfold도 쉽게 사용 가능할 것이라 예상.
    -> Test 하는 중 
- x_train, x_val 을 나누는 경우 seed 와 random_state 로 y를 맞출 수 있지 않을까??

- 이 경우 fit_generator를 사용하지 못할 것이라고 예상이 된다.
  np.save를 이용하여 x_train을 저장한 후 np.load를 사용해보자 (keras66_3,4 참고)


# flow_from_directory 에 y를 넣어서 한번에 generator를 만들 수 있을까??

- classes = y 가능한가?? 
    -> X
'''
# Testing
x_train = np.load('./dacon3/data/x_train_merge_1.npy')
for i in range(1,10):
    a = np.load('./dacon3/data/x_train_merge_{}.npy'.format(i+1))
    print(a.shape)
    x_train = np.append(x_train, a, axis=0)
print(x_train.shape)

x_train = x_train.reshape(50000, 256, 256, 1)
print(x_train.shape)

train_datagen = ImageDataGenerator(
    rescale=1./255,)
train_generator = train_datagen.flow_from_directory(
    '../data', 
    classes=['dirty_mnist_2nd'],
    batch_size=50000, # batch_size 수정해서 fit or fit_generator에 사용할 것.
    target_size=(256, 256), 
    color_mode='grayscale',
    shuffle=False,
    class_mode=None)

for i in train_generator:
    print(type(i))
    print(i.shape)

    print(np.array_equal(i, x_train))
    break