# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.compat.v1.set_random_seed(66)

from tensorflow.keras.datasets import cifar10

tf.compat.v1.disable_eager_execution()
print(tf.executing_eagerly())   # False

print(tf.__version__)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

x_train = x_train.reshape(50000, 32, 32, 3).astype('float32')/255
x_test = x_test.reshape(10000, 32, 32, 3).astype('float32')/255

training_epochs = 100
batch_size = 10000
total_batch = int(len(x_train) / batch_size)

x = tf.compat.v1.placeholder(tf.float32, shape=[None,32,32,3])
y = tf.compat.v1.placeholder(tf.float32, shape=[None,10])

w1 = tf.compat.v1.get_variable('w1', shape=[2,2,3,16])
L1 = tf.nn.conv2d(x, w1, strides=[1,1,1,1], padding='SAME')
L1 = tf.nn.relu(L1)
# L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

w2 = tf.compat.v1.get_variable('w2', shape=[2,2,16,16])
L2 = tf.nn.conv2d(L1, w2, strides=[1,1,1,1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# w3 = tf.compat.v1.get_variable('w3', shape=[2,2,16,32])
# L3 = tf.nn.conv2d(L2, w3, strides=[1,1,1,1], padding='SAME')
# L3 = tf.nn.relu(L3)
# # L3 = tf.nn.max_pool(L3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# w4 = tf.compat.v1.get_variable('w4', shape=[2,2,32,32])
# L4 = tf.nn.conv2d(L3, w4, strides=[1,1,1,1], padding='SAME')
# L4 = tf.nn.relu(L4)
# L4 = tf.nn.max_pool(L4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

L_Flatten = tf.reshape(L2, [-1, 16*16*16])

w5 = tf.compat.v1.get_variable('w5', shape=[16*16*16, 32])
b5 = tf.Variable(tf.random.normal([32]), name='bias1')
L5 = tf.nn.relu(tf.matmul(L_Flatten, w5) + b5)

# w6 = tf.compat.v1.get_variable('w6', shape=[64, 32])
# b6 = tf.Variable(tf.random.normal([32]), name='bias2')
# L6 = tf.nn.relu(tf.matmul(L5, w6) + b6)

w7 = tf.compat.v1.get_variable('w7', shape=[32, 10])
b7 = tf.Variable(tf.random.normal([10]), name='bias3')
hypothesis = tf.nn.relu(tf.matmul(L5, w7) + b7)


loss = tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(hypothesis)))
optimizer = tf.compat.v1.train.AdamOptimizer(0.001).minimize(loss)

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0

        for i in range(total_batch):
            start = i * batch_size
            end = start * batch_size

            batch_x, batch_y = x_train[start:end], y_train[start:end]

            feed_dict = {x:batch_x, y:batch_y}

            sess.run(optimizer, feed_dict=feed_dict)

            avg_cost += sess.run(loss, feed_dict) / total_batch

        print('Epoch :', (epoch + 1), '\tloss : {:.9f}'.format(avg_cost))
        
        del batch_x, batch_y

    print('Finish !!')

    # for i in range()
    prediction = tf.equal(tf.math.argmax(hypothesis, 1), tf.math.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
    print('Acc :', sess.run(accuracy, feed_dict={x:x_test, y:y_test}))