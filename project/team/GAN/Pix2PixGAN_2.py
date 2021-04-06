import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

PATH = 'G:/공유 드라이브/Team_project/01_data/'

BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 1280
IMG_HEIGHT = 720

def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)
    
    image = tf.cast(image, tf.float32)
    
    return image

def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image

def random_crop(input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image[0], cropped_image[1]

# normalizing the images to [-1, 1]
def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image

@tf.function()
def random_jitter(input_image, real_image):
    # resizing to 286 x 286 x 3
    input_image, real_image = resize(input_image, real_image, 800, 1400)

    # randomly cropping to 256 x 256 x 3
    input_image, real_image = random_crop(input_image, real_image)

    if tf.random.uniform(()) > 0.5:
        # random mirroring
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image

def load_image_train(image_file, real_image):
    input_image, real_image = load(image_file), load(real_image)
    input_image, real_image = random_jitter(input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image

def load_image_test(image_file, real_image):
    input_image, real_image = load(image_file), load(real_image)
    input_image, real_image = resize(input_image, real_image, IMG_HEIGHT, IMG_WIDTH)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image

# train_data
input_img = tf.data.Dataset.list_files(PATH + 'padding_img/*.jpg', shuffle=False)
output_img = tf.data.Dataset.list_files(PATH + 'resize_img/*.jpg', shuffle=False)
print(input_img)
print(output_img)

train_dataset = tf.data.Dataset.zip((input_img, output_img))
print(train_dataset)

train_dataset = train_dataset.map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

# test_data
input_img = tf.data.Dataset.list_files(PATH + 'test_img/padding_img/*.jpg', shuffle=False)
output_img = tf.data.Dataset.list_files(PATH + 'test_img/resize_img/*.jpg', shuffle=False)

test_dataset = tf.data.Dataset.zip((input_img, output_img))

test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.shuffle(BUFFER_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

del input_img
del output_img

_, ax = plt.subplots(4, 2, figsize=(10, 15))
for i, (example_input, example_target) in enumerate(test_dataset.take(4)):
    ax[i, 0].imshow(example_input[0])
    ax[i, 1].imshow(example_target[0])
    ax[i, 0].set_title("Input image")
    ax[i, 0].set_title("Input image")
    ax[i, 1].set_title("Translated image")
    ax[i, 0].axis("off")
    ax[i, 1].axis("off")
plt.tight_layout()
plt.show()




# Modeling
class ReflectionPadding2D(layers.Layer):
    """Implements Reflection Padding as a layer.

    Args:
        padding(tuple): Amount of padding for the
        spatial dimensions.

    Returns:
        A padded tensor with the same type as the input tensor.
    """

    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        padding_tensor = [
            [0, 0],
            [padding_height, padding_height],
            [padding_width, padding_width],
            [0, 0],
        ]
        return tf.pad(input_tensor, padding_tensor, mode="REFLECT")

kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

def residual_block(
    x,
    activation,
    kernel_initializer=kernel_init,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding="valid",
    gamma_initializer=gamma_init,
    use_bias=False,
):
    dim = x.shape[-1]
    input_tensor = x

    x = ReflectionPadding2D()(input_tensor)
    x = layers.Conv2D(
        dim,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = layers.BatchNormalization(gamma_initializer=gamma_initializer)(x)
    x = activation(x)

    x = ReflectionPadding2D()(x)
    x = layers.Conv2D(
        dim,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = layers.BatchNormalization(gamma_initializer=gamma_initializer)(x)
    x = layers.add([input_tensor, x])
    return x


def downsample(
    x,
    filters,
    activation,
    kernel_initializer=kernel_init,
    kernel_size=(3, 3),
    strides=(2, 2),
    padding="same",
    gamma_initializer=gamma_init,
    use_bias=False,
):
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = layers.BatchNormalization(gamma_initializer=gamma_initializer)(x)
    if activation:
        x = activation(x)
    return x


def upsample(
    x,
    filters,
    activation,
    kernel_size=(3, 3),
    strides=(2, 2),
    padding="same",
    kernel_initializer=kernel_init,
    gamma_initializer=gamma_init,
    use_bias=False,
):
    x = layers.Conv2DTranspose(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        kernel_initializer=kernel_initializer,
        use_bias=use_bias,
    )(x)
    x = layers.BatchNormalization(gamma_initializer=gamma_initializer)(x)
    if activation:
        x = activation(x)
    return x


# GENERATOR
def get_generator(basic_filters=64,kernel_size=4,drop_out=0.5,alpha=0.2,name=None):
    
    initializer = tf.random_normal_initializer(0.,0.02)
    inputs = layers.Input(shape=(720,1280,3), name=name + "_img_input")
    layer1 = layers.Conv2D(filters = basic_filters,kernel_size=4,strides=2,padding='same',use_bias=False,kernel_initializer=initializer)(inputs)
    layer1 = layers.LeakyReLU()(layer1)
    layer1_ = layer1
    
    layer2 = layers.Conv2D(filters=basic_filters*2,kernel_size=kernel_size,strides=2,padding='same',use_bias=False,kernel_initializer=initializer)(layer1)
    layer2_ = layers.BatchNormalization()(layer2)
    layer2 = layers.LeakyReLU()(layer2_)
    
    layer3 = layers.Conv2D(filters=basic_filters*4,kernel_size=kernel_size,strides=2,padding='same',use_bias=False,kernel_initializer=initializer)(layer2)
    layer3_ = layers.BatchNormalization()(layer3)
    layer3 = layers.LeakyReLU()(layer3_)
    
    layer4 = layers.Conv2D(filters=basic_filters*8,kernel_size=kernel_size,strides=(2,2),padding='same',use_bias=False,kernel_initializer=initializer)(layer3)
    layer4_ = layers.BatchNormalization()(layer4)
    layer4 = layers.LeakyReLU()(layer4_)
    
    layer5 = layers.Conv2D(filters=basic_filters*8,kernel_size=kernel_size,strides=(3,2),padding='same',use_bias=False,kernel_initializer=initializer)(layer4)
    layer5_ = layers.BatchNormalization()(layer5)
    layer5 = layers.LeakyReLU()(layer5_)
    
    layer6 = layers.Conv2D(filters=basic_filters*8,kernel_size=kernel_size,strides=(3,2),padding='same',use_bias=False,kernel_initializer=initializer)(layer5)
    layer6_ = layers.BatchNormalization()(layer6)
    layer6 = layers.LeakyReLU()(layer6_)
    
    layer7 = layers.Conv2D(filters=basic_filters*8,kernel_size=kernel_size,strides=(1,2),padding='same',use_bias=False,kernel_initializer=initializer)(layer6)
    layer7_ = layers.BatchNormalization()(layer7)
    layer7 = layers.LeakyReLU()(layer7_)
    
    layer8 = layers.Conv2D(filters=basic_filters*16,kernel_size=kernel_size,strides=(1,2),padding='same',use_bias=False,kernel_initializer=initializer)(layer7)
    layer8_ = layers.BatchNormalization()(layer8)
    layer8 = layers.LeakyReLU()(layer8_)
    
    # 가운데
    layer9 = layers.Conv2D(filters=basic_filters*16,kernel_size=kernel_size,strides=(5,5),padding='same',use_bias=False,kernel_initializer=initializer)(layer8)
    layer9_ = layers.BatchNormalization()(layer9)
    layer9 = layers.LeakyReLU()(layer9_)
    # 가운데
    
    layer10 = layers.Conv2DTranspose(filters=basic_filters*16,kernel_size=kernel_size,strides=(5,5),padding='same',kernel_initializer=initializer,use_bias=False)(layer9)
    layer10 = layers.BatchNormalization()(layer10)
    layer10 = layer10+layer8_
    layer10 = layers.Dropout(drop_out)(layer10)
    layer10 = layers.ReLU()(layer10)
    
    layer11 = layers.Conv2DTranspose(filters=basic_filters*8,kernel_size=kernel_size,strides=(1,2),padding='same',kernel_initializer=initializer,use_bias=False)(layer10)
    layer11 = layers.BatchNormalization()(layer11)
    layer11 = layer11+layer7_
    layer11 = layers.Dropout(drop_out)(layer11)
    layer11 = layers.ReLU()(layer11)
    
    layer12 = layers.Conv2DTranspose(filters=basic_filters*8,kernel_size=kernel_size,strides=(1,2),padding='same',kernel_initializer=initializer,use_bias=False)(layer11)
    layer12 = layers.BatchNormalization()(layer12)
    layer12 = layer12+layer6_
    layer12 = layers.Dropout(drop_out)(layer12)
    layer12 = layers.ReLU()(layer12)
    
    layer13 = layers.Conv2DTranspose(filters=basic_filters*8,kernel_size=kernel_size,strides=(3,2),padding='same',kernel_initializer=initializer,use_bias=False)(layer12)
    layer13 = layers.BatchNormalization()(layer13)
    layer13 = layer13+layer5_
    layer13 = layers.Dropout(drop_out)(layer13)
    layer13 = layers.ReLU()(layer13)
    
    layer14 = layers.Conv2DTranspose(filters=basic_filters*8,kernel_size=kernel_size,strides=(3,2),padding='same',kernel_initializer=initializer,use_bias=False)(layer13)
    layer14 = layers.BatchNormalization()(layer14)
    layer14 = layer14+layer4_
    layer14 = layers.Dropout(drop_out)(layer14)
    layer14 = layers.ReLU()(layer14)
    
    layer15 = layers.Conv2DTranspose(filters=basic_filters*4,kernel_size=kernel_size,strides=(2,2),padding='same',kernel_initializer=initializer,use_bias=False)(layer14)
    layer15 = layers.BatchNormalization()(layer15)
    layer15 = layer15+layer3_
    layer15 = layers.Dropout(drop_out)(layer15)
    layer15 = layers.ReLU()(layer15)
    
    layer16 = layers.Conv2DTranspose(filters=basic_filters*2,kernel_size=kernel_size,strides=(2,2),padding='same',kernel_initializer=initializer,use_bias=False)(layer15)
    layer16 = layers.BatchNormalization()(layer16)
    layer16 = layer16+layer2_
    layer16 = layers.Dropout(drop_out)(layer16)
    layer16 = layers.ReLU()(layer16)
    
    layer17 = layers.Conv2DTranspose(filters=basic_filters,kernel_size=kernel_size,strides=(2,2),padding='same',kernel_initializer=initializer,use_bias=False)(layer16)
    layer17 = layers.BatchNormalization()(layer17)
    layer17 = layer17+layer1_
    layer17 = layers.Dropout(drop_out)(layer17)
    layer17 = layers.ReLU()(layer17)
    
    outputs_ = layers.Conv2DTranspose(filters=3,kernel_size=kernel_size,strides=2,padding='same',kernel_initializer=initializer,use_bias=False,activation='tanh')(layer17)
    outputs = (1-alpha)*outputs_+alpha*inputs
    
    outputs = layers.Conv2DTranspose(3,4,strides=1,padding='same',kernel_initializer=initializer,activation='tanh')(outputs)    
    
    model = keras.models.Model(inputs=inputs,outputs=outputs)
    
    return model

'''
def get_generator(
    filters=64,
    num_downsampling_blocks=2,
    num_residual_blocks=9,
    num_upsample_blocks=2,
    gamma_initializer=gamma_init,
    name=None,
):
    img_input = layers.Input(shape=(720,1280,3), name=name + "_img_input")
    x = ReflectionPadding2D(padding=(3, 3))(img_input)
    x = layers.Conv2D(filters, (7, 7), kernel_initializer=kernel_init, use_bias=False)(
        x
    )
    x = layers.BatchNormalization(gamma_initializer=gamma_initializer)(x)
    x = layers.Activation("relu")(x)

    # Downsampling
    for _ in range(num_downsampling_blocks):
        filters *= 2
        x = downsample(x, filters=filters, activation=layers.Activation("relu"))

    # Residual blocks
    for _ in range(num_residual_blocks):
        x = residual_block(x, activation=layers.Activation("relu"))

    # Upsampling
    for _ in range(num_upsample_blocks):
        filters //= 2
        x = upsample(x, filters, activation=layers.Activation("relu"))

    # Final block
    x = ReflectionPadding2D(padding=(3, 3))(x)
    x = layers.Conv2D(3, (7, 7), padding="valid")(x)
    x = layers.Activation("tanh")(x)

    model = keras.models.Model(img_input, x, name=name)
    return model
'''

# DISCRIMINATOR
def get_discriminator(
    filters=64, kernel_initializer=kernel_init, num_downsampling=5, name=None
):
    img_input = layers.Input(shape=(720,1280,3), name=name + "_img_input")
    x = layers.Conv2D(
        filters,
        (4, 4),
        strides=(2, 2),
        padding="same",
        kernel_initializer=kernel_initializer,
    )(img_input)
    x = layers.LeakyReLU(0.2)(x)

    num_filters = filters
    for num_downsample_block in range(5):
        num_filters *= 2
        if num_downsample_block < 4:
            x = downsample(
                x,
                filters=num_filters,
                activation=layers.LeakyReLU(0.2),
                kernel_size=(4, 4),
                strides=(2, 2),
            )
        else:
            x = downsample(
                x,
                filters=num_filters,
                activation=layers.LeakyReLU(0.2),
                kernel_size=(4, 4),
                strides=(1, 1),
            )

    x = layers.Conv2D(
        1, (4, 4), strides=(1, 1), padding="same", kernel_initializer=kernel_initializer
    )(x)

    model = keras.models.Model(inputs=img_input, outputs=x, name=name)
    return model

# Get the generators
gen_G = get_generator(name="generator_G")
# gen_F = get_generator(name="generator_F")

# Get the discriminators
# disc_X = get_discriminator(name="discriminator_X")
disc_Y = get_discriminator(name="discriminator_Y")

gen_G.summary()
print(gen_G)
# print(gen_F)
disc_Y.summary()
# print(disc_X)
print(disc_Y)




# Build the Pix2Pix model
class Pix2Pix(keras.Model):
    def __init__(
        self,
        generator_G,
        discriminator_Y,
        LAMBDA=100
    ):
        super(Pix2Pix, self).__init__()
        self.gen_G = generator_G
        self.disc_Y = discriminator_Y
        self.LAMBDA = LAMBDA

    def compile(
        self,
        gen_G_optimizer,
        disc_Y_optimizer,
        gen_loss_fn,
        disc_loss_fn,
    ):
        super(Pix2Pix, self).compile()
        self.gen_G_optimizer = gen_G_optimizer
        self.disc_Y_optimizer = disc_Y_optimizer
        self.generator_loss_fn = gen_loss_fn
        self.discriminator_loss_fn = disc_loss_fn

    def train_step(self, batch_data):
        real_x, real_y = batch_data
        Lambda = self.LAMBDA
        with tf.GradientTape(persistent=True) as gen_tape, tf.GradientTape(persistent=True) as disc_tape:
            # Generate fake image
            fake_y = self.gen_G(real_x, training=True)

            # Discriminator output
            disc_real_y = self.disc_Y(real_y, training=True)
            disc_fake_y = self.disc_Y(fake_y, training=True)

            # Generator adverserial loss
            gen_G_loss = self.generator_loss_fn(disc_fake_y, fake_y, real_y, Lambda)

            # Discriminator loss
            disc_Y_loss = self.discriminator_loss_fn(disc_real_y, disc_fake_y)

        # Get the gradients for the generators
        grads_G = gen_tape.gradient(gen_G_loss, self.gen_G.trainable_variables)

        # Get the gradients for the discriminators
        disc_Y_grads = disc_tape.gradient(disc_Y_loss, self.disc_Y.trainable_variables)

        # Update the weights of the generators
        self.gen_G_optimizer.apply_gradients(
            zip(grads_G, self.gen_G.trainable_variables)
        )
        # Update the weights of the discriminators
        self.disc_Y_optimizer.apply_gradients(
            zip(disc_Y_grads, self.disc_Y.trainable_variables)
        )

        return {
            "G_loss": gen_G_loss,
            "D_Y_loss": disc_Y_loss,
        }



# Loss function for evaluating adversarial loss
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Define the loss function for the generators
def generator_loss_fn(disc_generated_output, gen_output, target, LAMBDA):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss

# Define the loss function for the discriminators
def discriminator_loss_fn(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss


# Create pix2pix gan model
pix2pix_gan_model = Pix2Pix(
    generator_G=gen_G, discriminator_Y=disc_Y
)

# Compile the model
pix2pix_gan_model.compile(
    gen_G_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    disc_Y_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    gen_loss_fn=generator_loss_fn,
    disc_loss_fn=discriminator_loss_fn,
)
# Callbacks
checkpoint_filepath = "./project/team/data/model_checkpoints"
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='G_loss',
    mode='auto',
    save_best_only=True,
    verbose=1
)
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='G_loss',
    factor=0.9,
    patience=8, 
    mode='auto',
    verbose=1
)
early_stopping = keras.callbacks.EarlyStopping(
    monitor='G_loss',
    patience=30,
    mode='auto'
)
# Here we will train the model for just one epoch as each epoch takes around
# 7 minutes on a single P100 backed machine.
pix2pix_gan_model.fit(
    train_dataset,
    epochs=1000,
    callbacks=[model_checkpoint_callback, reduce_lr, early_stopping],
)




# Predict
weight_file = './project/team/data/model_checkpoints'
pix2pix_gan_model.load_weights(weight_file).expect_partial()
print("Weights loaded successfully")

_, ax = plt.subplots(4, 2, figsize=(10, 15))
for i, (example_input, example_target) in enumerate(test_dataset.take(4)):
    prediction = pix2pix_gan_model.gen_G(example_input, training=False)[0].numpy()
    prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
    example_input = (example_input[0] * 127.5 + 127.5).numpy().astype(np.uint8)

    ax[i, 0].imshow(example_input)
    ax[i, 1].imshow(prediction)
    ax[i, 0].set_title("Input image")
    ax[i, 0].set_title("Input image")
    ax[i, 1].set_title("Translated image")
    ax[i, 0].axis("off")
    ax[i, 1].axis("off")

    prediction = keras.preprocessing.image.array_to_img(prediction)
    # prediction.save("predicted_img_{i}.png".format(i=i))
plt.tight_layout()
plt.show()
