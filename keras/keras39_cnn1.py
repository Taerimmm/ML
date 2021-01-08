'''
Conv2D(10, (3,3), input_shape=(5,5,1))
명칭 : filters, kernel_size, data_format(batch_size, row, column, channel)
'''

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D

model = Sequential()
model.add(Conv2D(filters=10, kernel_size=(2,2), strides=1,
                 padding='same', input_shape=(10,10,3))) # 4차원 받아서 4차원 출력
# padding='smae' : 원래 shape랑 같게 해준다.
# padding의 default는 'valid'
# stride : 간격
model.add(MaxPooling2D(pool_size=(2,3)))
# 작은 원소들을 제거하는 작업 수행
model.add(Conv2D(9, (2,2), padding='valid'))
# model.add(Conv2D(9, (2,3)))
# model.add(Conv2D(8, 2)) # 2를 (2,2)로 인식
model.add(Flatten()) # 2차원으로 바꿔주는 역할
model.add(Dense(1))

model.summary()

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 9, 9, 10)          50
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 8, 8, 9)           369
_________________________________________________________________
flatten (Flatten)            (None, 576)               0
_________________________________________________________________
dense (Dense)                (None, 1)                 577
=================================================================
Total params: 996
Trainable params: 996
Non-trainable params: 0
_________________________________________________________________

number_parameters = out_channels * (in_channels * kernel_height * kernel_width + 1)

50 = 10 * (1 * 2 * 2 + 1)
369 = 9 * (10 * 2 * 2 + 1)

'''