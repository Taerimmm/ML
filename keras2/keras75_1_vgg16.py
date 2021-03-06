from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
# include_top : False로 해야 input_shape를 원하는 사이즈로 가능
print(vgg16.weights)

vgg16.trainable = False

vgg16.summary()
print(len(vgg16.weights))
print(len(vgg16.trainable_weights))

'''
Total params: 14,714,688
Trainable params: 0
Non-trainable params: 14,714,688
'''

vgg16.trainable = True

vgg16.summary()
print(len(vgg16.weights))
print(len(vgg16.trainable_weights))

'''
Total params: 14,714,688
Trainable params: 14,714,688
Non-trainable params: 0
'''