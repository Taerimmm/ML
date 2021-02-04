from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(x_train.shape, x_test.shape)  # (50000, 32, 32, 3) (10000, 32, 32, 3)
print(y_train.shape, y_test.shape)  # (50000, 10) (10000, 10)

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

datagen.fit(x_train)


# datagen.flow 의 출력 형식 확인

print(x_train[:2].shape)
# print(datagen.apply_transform)
print(y_train[:5])
count = 0
for x_gen, y_gen in datagen.flow(x_train[:5], y_train[:5], batch_size=2):
    print(x_gen)
    print(x_gen[0][0])
    print(y_gen)
    print(x_gen.shape)
    print(y_gen.shape)
    print()
    if count > 5:
        break
    count += 1
print(count)

'''
x_train은 datagen.fit으로 변형이 되었고
1 loop마다 batch_size만큼 x_train, y_train에서 추출해낸다
'''

model = Sequential()
model.add(Conv2D(64, 3, padding='same', activation='relu', input_shape=x_train.shape[1:]))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
epochs = 1
model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
          steps_per_epoch=len(x_train) / 32, epochs = epochs)

######## 위의 fit_generator 랑 밑의 이중 for 문이랑 동일 (예상) ################
''' 
fit_generator 에서 steps_per_epoch의 Default는 None이고 이때 steps_per_epoch = len(generator) 이 된다.
len(generator) = len(x_train) / datagen.flow의 batch_size
이와 같으면 1 epo에서 뽑아내는 data의 총 개수는 x_train의 총 개수랑 같아지게 된다.

따라서 fit_generator의 1 epo는 이중 for 문에서의 바깥 for 문 1 loop와 같다.
또한 fit이 중첩되어 훈련이 누적된다
'''

my_batch_size=32
for e in range(epochs):
    print('Epoch', e)
    batches = 0
    for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=my_batch_size):
        print(x_batch.shape, y_batch.shape)
        model.fit(x_batch, y_batch, epochs=epochs)
        batches += 1

        if batches >= len(x_train) / my_batch_size:
            break

    print('Epoch', e+1, 'end !!!!')

    if e == 0:
        break
