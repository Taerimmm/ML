import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Hyperparameter
img_rows = 28
img_cols = 28
channels = 1

img_shape = (img_rows, img_cols, channels)

z_dim = 100

# 생성자
def build_generator(img_shape, z_dim):
    model = Sequential()

    model.add(Dense(128, input_dim=z_dim))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(28 * 28 * 1, activation='tanh'))
    model.add(Reshape(img_shape))

    return model

# 판별자
def build_discriminator(img_shape):
    model = Sequential()
    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(1, activation='sigmoid'))
    
    return model

# GAN 모델 생성
def build_gan(generator, discriminator):
    model = Sequential()

    model.add(generator) # 생성자 + 판별자 모델 연결
    model.add(discriminator)

    return model

discriminator = build_discriminator(img_shape)
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

generator = build_generator(img_shape, z_dim)

discriminator.trainable = False # 생성자 훈련할 때 판별자 파라미터 동결하기

gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer=Adam())

gan.summary()

# GAN 훈련

losses = []
accuracies = []
iteration_checkpoints = []

def train(iterations, batch_size, sample_interval):
    (x_train, _), (_, _) = mnist.load_data()

    # 흑백 픽셀 값들을 (0,255) 에서 (-1,1) 사이로 조정
    x_train = x_train / 127.5 - 1.0 
    x_train = np.expand_dims(x_train, axis=3)

    # 진짜 이미지 레이블 : 모두 1
    real = np.ones((batch_size, 1))
    # 가짜 이미지 레이블 : 모두 0
    fake = np.zeros((batch_size, 1))

    for iteration in range(iterations):

        # -------------------------
        #  판별자 훈련
        # -------------------------

        # 진짜 이미지에서 랜덤 배치 가져오기
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        imgs = x_train[idx]

        # 가짜 이미지 배치 생성
        z = np.random.normal(0, 1, (batch_size, 100))
        gen_imgs = generator.predict(z)
        
        # 판별자 훈련
        d_loss_real = discriminator.train_on_batch(imgs, real)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss, accuray = 0.5 * np.add(d_loss_real, d_loss_fake)
        # print(d_loss, accuray)

        # ---------------------
        #  생성자 훈련
        # ---------------------

        # 가짜 이미지 배치 생성
        z = np.random.normal(0, 1, (batch_size, 100))
        gen_imgs = generator.predict(z)

        # 생성자 훈련
        g_loss = gan.train_on_batch(z, real)

        if (iteration + 1) % sample_interval == 0:
            losses.append((d_loss, g_loss))
            accuracies.append(100 * accuray)
            iteration_checkpoints.append(iteration + 1)

            print("% d [D 손실 : %f, 정확도 : %.2f%%] [G 손실 : %f]" % (iteration + 1, d_loss, 100.0 * accuray, g_loss))

            sample_images(generator)
            
        if (iteration % 5000) == 0:
            plt.show()

# 생성된 이미지 출력
def sample_images(generator, image_grid_rows=4, image_grid_columns=4):
    # 랜덤한 잡음 샘플링
    z = np.random.normal(0, 1, (image_grid_rows * image_grid_columns, z_dim))
    # 랜덤한 잡음에서 이미지 생성
    gen_imgs = generator.predict(z)
    # 이미지 픽셀 값을 (0,1) 사이로 조정
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(image_grid_rows, image_grid_columns, figsize=(4,4), sharey=True, sharex=True)

    cnt = 0
    for i in range(image_grid_rows):
        for j in range(image_grid_columns):
            axs[i,j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1

# 모델 실행
iterations = 20000
batch_size = 128
sample_interval = 1000

train(iterations, batch_size, sample_interval)
