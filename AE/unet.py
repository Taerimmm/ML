import tensorflow as tf
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Conv2D,BatchNormalization,Dropout,Activation,LeakyReLU,UpSampling2D,Input,Dense,Reshape,Flatten,Conv2DTranspose,ReLU,concatenate,ZeroPadding2D
import numpy as np
def BaseUnet_modeling(kernel_size=4,dropout=0.5):
    initializer = tf.random_normal_initializer(0.,0.02)
    inputs = Input(shape=(256,256,3))
    layer1 = Conv2D(filters=64,kernel_size=kernel_size,strides=2,padding='same',use_bias=False,kernel_initializer=initializer)(inputs)
    layer1 = LeakyReLU()(layer1)
    layer1_ = layer1

    layer2 = Conv2D(filters=128,kernel_size=kernel_size,strides=2,padding='same',use_bias=False,kernel_initializer=initializer)(layer1)
    layer2_ = BatchNormalization()(layer2)
    layer2 = LeakyReLU()(layer2_)

    layer3 = Conv2D(filters=256,kernel_size=kernel_size,strides=2,padding='same',use_bias=False,kernel_initializer=initializer)(layer2)
    layer3_ = BatchNormalization()(layer3)
    layer3 = LeakyReLU()(layer3_)

    layer4 = Conv2D(filters=512,kernel_size=kernel_size,strides=2,padding='same',use_bias=False,kernel_initializer=initializer)(layer3)
    layer4_ = BatchNormalization()(layer4)
    layer4 = LeakyReLU()(layer4_)

    layer5 = Conv2D(filters=512,kernel_size=kernel_size,strides=2,padding='same',use_bias=False,kernel_initializer=initializer)(layer4)
    layer5_ = BatchNormalization()(layer5)
    layer5 = LeakyReLU()(layer5_)

    layer6 = Conv2D(filters=512,kernel_size=kernel_size,strides=2,padding='same',use_bias=False,kernel_initializer=initializer)(layer5)
    layer6_ = BatchNormalization()(layer6)
    layer6 = LeakyReLU()(layer6_)

    layer7 = Conv2D(filters=512,kernel_size=kernel_size,strides=2,padding='same',use_bias=False,kernel_initializer=initializer)(layer6)
    layer7_ = BatchNormalization()(layer7)
    layer7 = LeakyReLU()(layer7_)



    layer8 = Conv2D(filters=512,kernel_size=kernel_size,strides=2,padding='same',use_bias=False,kernel_initializer=initializer)(layer7)
    layer8_ = BatchNormalization()(layer8)
    layer8 = LeakyReLU()(layer8_)



    layer9 = Conv2DTranspose(filters=512,kernel_size=kernel_size,strides=2,padding='same',kernel_initializer=initializer,use_bias=False)(layer8)
    layer9 = BatchNormalization()(layer9)
    layer9 = layer9+layer7_
    layer9 = Dropout(dropout)(layer9)
    layer9 = ReLU()(layer9)

    layer10 = Conv2DTranspose(filters=512,kernel_size=kernel_size,strides=2,padding='same',kernel_initializer=initializer,use_bias=False)(layer9)
    layer10 = BatchNormalization()(layer10)
    layer10 = concatenate([layer10,layer6_])
    layer10 = Dropout(dropout)(layer10)
    layer10 = ReLU()(layer10)

    layer11 = Conv2DTranspose(filters=512,kernel_size=kernel_size,strides=2,padding='same',kernel_initializer=initializer,use_bias=False)(layer10)
    layer11 = BatchNormalization()(layer11)
    layer11 = layer11+layer5_
    layer11 = Dropout(dropout)(layer11)
    layer11 = ReLU()(layer11)

    layer12 = Conv2DTranspose(filters=512,kernel_size=kernel_size,strides=2,padding='same',kernel_initializer=initializer,use_bias=False)(layer11)
    layer12 = BatchNormalization()(layer12)
    layer12 = layer12+layer4_
    layer12 = ReLU()(layer12)

    layer13 = Conv2DTranspose(filters=256,kernel_size=kernel_size,strides=2,padding='same',kernel_initializer=initializer,use_bias=False)(layer12)
    layer13 = BatchNormalization()(layer13)
    layer13 = layer13+layer3_
    layer13 = ReLU()(layer13)

    layer14 = Conv2DTranspose(filters=128,kernel_size=kernel_size,strides=2,padding='same',kernel_initializer=initializer,use_bias=False)(layer13)
    layer14 = BatchNormalization()(layer14)
    layer14 = layer14+layer2_
    layer14 = ReLU()(layer14)

    layer15 = Conv2DTranspose(filters=64,kernel_size=kernel_size,strides=2,padding='same',kernel_initializer=initializer,use_bias=False)(layer14)
    layer15 = BatchNormalization()(layer15)
    layer15 = layer15+layer1_
    layer15 = ReLU()(layer15)

    outputs = Conv2DTranspose(3,4,strides=2,padding='same',kernel_initializer=initializer,activation='tanh')(layer15)

    Generator = Model(inputs=inputs,outputs=outputs)
    return Generator
model = BaseUnet_modeling()
model.summary()