import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(66)

# 1. 데이터
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255

learning_rate = 0.001
training_epochs = 15
batch_size = 100
total_batch = int(len(x_train) / batch_size)

x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, 10])

# 2. 모델구성

# L1.
w1 = tf.get_variable('w1', shape=[3, 3, 1, 32])             # 3, 3 : kernel_size , 1 : color , 32 : out_channel
L1 = tf.nn.conv2d(x, w1, strides=[1,1,1,1], padding='SAME') # strides=[1,2,2,1] <- stirdes : 2
print(L1)
# Conv2D(filter, kernel_size, input_shape)
# Conv2D(32, (3,3), input_shape=(28,28,1))
L1 = tf.nn.selu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(L1)       # (?, 14, 14, 32)

# L2.
w2 = tf.get_variable('w2', shape=[3, 3, 32, 64])             
L2 = tf.nn.conv2d(L1, w2, strides=[1,1,1,1], padding='SAME') 
L2 = tf.nn.elu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(L2)       # (?, 7, 7, 64)


# L3.
w3 = tf.get_variable('w3', shape=[3, 3, 64, 128])             
L3 = tf.nn.conv2d(L2, w3, strides=[1,1,1,1], padding='SAME') 
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(L3)       # (?, 4, 4, 128)

# L4.
w4 = tf.get_variable('w4', shape=[3, 3, 128, 64])             
L4 = tf.nn.conv2d(L3, w4, strides=[1,1,1,1], padding='SAME') 
L4 = tf.nn.selu(L4)
L4 = tf.nn.max_pool(L4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(L4)       # (?, 2, 2, 64)

# Flatten
L_flat = tf.reshape(L4, [-1, 2*2*64])
print('Flattne :', L_flat)

# L5.
w5 = tf.get_variable('w5', shape=[2*2*64, 64], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([64]), name= 'b1')
L5 = tf.nn.selu(tf.matmul(L_flat, w5) + b5)
L5 = tf.nn.dropout(L5, keep_prob=0.2)
print(L5)

# L6.
w6 = tf.get_variable('w6', shape=[64, 32], initializer=tf.contrib.layers.xavier_initializer())
b6 = tf.Variable(tf.random_normal([32]), name= 'b2')
L6 = tf.nn.selu(tf.matmul(L5, w6) + b6)
L6 = tf.nn.dropout(L6, keep_prob=0.2)
print(L6)

# L7.
w7 = tf.get_variable('w7', shape=[32, 10], initializer=tf.contrib.layers.xavier_initializer())
b7 = tf.Variable(tf.random_normal([10]), name= 'b2')
hypothesis = tf.nn.selu(tf.matmul(L6, w7) + b7)
print(hypothesis)

# 3. 컴파일, 훈련
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(hypothesis)))  
optimizer = tf.train.AdamOptimizer(0.0001).minimize(loss)

sess = tf.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epoch in range(training_epochs):
    avg_cost = 0

    for i in range(total_batch):
        start = i * batch_size
        end = start + batch_size

        batch_x, batch_y = x_train[start:end], y_train[start:end]

        feed_dict = {x:batch_x, y:batch_y}
        
        c, _ = sess.run([loss, optimizer], feed_dict=feed_dict)

        avg_cost += c / total_batch

    print('Epoch :', '%04d' %(epoch + 1),
          'cost = {:.9f}'.format(avg_cost))

print('Finish !!!')

prediction = tf.equal(tf.arg_max(hypothesis, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
print('Acc :', sess.run(accuracy, feed_dict={x:x_test, y:y_test}))

