from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
# include_top : False로 해야 input_shape를 원하는 사이즈로 가능
print(vgg16.weights)

vgg16.trainable = False

vgg16.summary()
print(len(vgg16.weights))               # 26
print(len(vgg16.trainable_weights))     # 0

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))

model.summary()

print("그냥 가중치의 수 :", len(model.weights))                             # 26 -> 32
print("동결하기 전 훈련되는 가중치의 수 :", len(model.trainable_weights))   # 0 -> 6
