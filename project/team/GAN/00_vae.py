# Variational AutoEncoder

from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from tensorflow.keras.datasets import mnist
import numpy as np

# Hyperparameter 설정
batch_size = 100
original_dim = 784
latent_dim = 2
intermediate_dim = 256
epochs = 50
epsilon_std = 1.0

# sampling 함수
def sampling(args: tuple):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=epsilon_std)

    return z_mean + K.exp(z_log_var / 2) * epsilon

# Encoder 만들기
x = Input(shape=(original_dim,), name='input')
h = Dense(intermediate_dim, activation='relu', name='encoding')(x)
z_mean = Dense(latent_dim, name='mean')(h)
z_log_var = Dense(latent_dim, name='log-variance')(h)

z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

encoder = Model(x, [z_mean, z_log_var, z], name='encoder')

# Decoder 만들기
input_decoder = Input(shape=(latent_dim,), name='decoder_input')
decoder_h = Dense(intermediate_dim, activation='relu', name='decoder_h')(input_decoder)
x_decoded = Dense(original_dim, activation='sigmoid', name='flat_decoded')(decoder_h)

decoder = Model(input_decoder, x_decoded, name='decoder')

# Model 결합
output_combined = decoder(encoder(x)[2])
vae = Model(x, output_combined)
vae.summary()

# 손실 함수 정의
kl_loss = -0.5 * K.sum(1 + z_log_var - K.exp(z_log_var) - K.square(z_mean), axis=-1)

vae.add_loss(K.mean(kl_loss) / 784.)

vae.compile(optimizer='rmsprop', loss='binary_crossentropy')

# 훈련 / 테스트 세트 분할
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# 훈련
vae.fit(x_train, x_train, shuffle=True, epochs=epochs, batch_size=batch_size)


# 테스트
output = vae.predict(x_test)

import matplotlib.pyplot as plt
import random
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = plt.subplots(2, 5, figsize=(20,7))

# 이미지 다섯 개를 무작위로 고른다.
random_images = random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다.

for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28,28), cmap='gray')
    
    if i == 0:
        ax.set_ylabel('INPUT', size=20)
    
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output[random_images[i]].reshape(28,28), cmap='gray')
    
    if i == 0:
        ax.set_ylabel('OUTPUT', size=20)
    
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()
