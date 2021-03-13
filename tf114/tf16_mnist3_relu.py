import tensorflow as tf
import numpy as np

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28*28).astype('float32')/255
x_test = x_test.reshape(10000, 28*28).astype('float32')/255

# 2. 모델 구성
x = tf.placeholder('float', [None,784])
y = tf.placeholder('float', [None,10])

w = tf.Variable(tf.random_normal([784,100]), name='weight1')
b = tf.Variable(tf.random_normal([100]), name='bias1')
# layer1 = tf.nn.softmax(tf.matmul(x,w) + b)
# layer1 = tf.nn.relu(tf.matmul(x,w) + b)
# layer1 = tf.nn.selu(tf.matmul(x,w) + b)
layer1 = tf.nn.elu(tf.matmul(x,w) + b)
layer1 = tf.nn.dropout(layer1, keep_prob=0.2) # dropout(0.2)

w2 = tf.Variable(tf.random_normal([100,50]), name='weight2')
b2 = tf.Variable(tf.random_normal([50]), name='bias2')
layer2 = tf.nn.relu(tf.matmul(layer1,w2) + b2)
layer2 = tf.nn.dropout(layer2, keep_prob=0.2) # dropout(0.2)

w3 = tf.Variable(tf.random_normal([50,10]), name='weight3')
b3 = tf.Variable(tf.random_normal([10]), name='bias3')
hypothesis = tf.nn.softmax(tf.matmul(layer2,w3) + b3)

# 3. 컴파일, 훈련
loss = tf.reduce_mean(- tf.reduce_sum(y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-4).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for step in range(2001):
        sess.run(optimizer, feed_dict={x:x_train, y:y_train})

        if step % 100 == 0: 
            print(step, '\tloss :', sess.run(loss, feed_dict={x:x_train, y:y_train}))