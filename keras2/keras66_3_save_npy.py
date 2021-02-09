import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest'
)                                                   # Train 에서는 Data가 많으면 좋기 때문에 증폭 사용
test_datagen = ImageDataGenerator(rescale=1./255)   # Test 에서는 Data를 증폭시킬 필요 X

# flow 또는 flow_from_directory
# 이미지 -> 데이터 화

# train_generator
xy_train = train_datagen.flow_from_directory(
    '../data/image/brain/train',
    target_size=(150,150),   # size 변경
    batch_size=160,          # batch_size 만큼 xy를 추출한다
    class_mode='binary'      # ad - Y 는 0 / normal - Y 는 1
)

# test_generator
xy_test = test_datagen.flow_from_directory(
    '../data/image/brain/test',
    target_size=(150,150),
    batch_size=120,
    class_mode='binary'     
)

print(xy_train)     # <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x000001E79C798550>
print(xy_test)      # <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x000001E79CD10B20>

print(xy_train[0])
print(xy_train[0][0])   
print(xy_train[0][0].shape)   # X -> (160, 150, 150, 3)
print(xy_train[0][1])  
print(xy_train[0][1].shape)   # Y -> (160,)

# save
np.save('../data/image/brain/npy/k66_train_x.npy', arr=xy_train[0][0])
np.save('../data/image/brain/npy/k66_train_y.npy', arr=xy_train[0][1])
np.save('../data/image/brain/npy/k66_test_x.npy', arr=xy_test[0][0])
np.save('../data/image/brain/npy/k66_test_y.npy', arr=xy_test[0][1])

x_train = np.load('../data/image/brain/npy/k66_train_x.npy')
y_train = np.load('../data/image/brain/npy/k66_train_y.npy')
x_test = np.load('../data/image/brain/npy/k66_test_x.npy')
y_test = np.load('../data/image/brain/npy/k66_test_y.npy')

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
