import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score

tf.compat.v1.set_random_seed(66)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print(x_train.shape, x_test.shape)  # (60000, 28, 28) (10000, 28, 28)
print(y_train.shape, y_test.shape)  # (60000,) (10000,)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])/255.
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])/255.

y_train = tf.keras.utils.to_categorical(y_train)

y_test = y_test.reshape(-1,1)

print(x_train.shape, x_test.shape)  # (60000, 784) (10000, 784)
print(y_train.shape, y_test.shape)  # (60000, 10) (10000, 1)

x = tf.placeholder(tf.float32, shape=[None,784])
y = tf.placeholder(tf.float32, shape=[None,10])

y_true= tf.placeholder(tf.float32, shape=[None,1])

w1 = tf.Variable(tf.random_normal([784,256],stddev=0.1), name='weight1')
b1 = tf.Variable(tf.random_normal([256],stddev=0.1), name='bias1')
layer1 = tf.nn.relu(tf.matmul(x,w1) + b1)

w2 = tf.Variable(tf.random_normal([256,64],stddev=0.1), name='weight2')
b2 = tf.Variable(tf.random_normal([64],stddev=0.1), name='bias2')
layer2 = tf.nn.relu(tf.matmul(layer1,w2) + b2)

w3 = tf.Variable(tf.random_normal([64,10],stddev=0.1), name='weight3')
b3 = tf.Variable(tf.random_normal([10],stddev=0.1), name='bias3')
hypothesis = tf.nn.softmax(tf.matmul(layer2,w3) + b3)

loss = tf.reduce_mean(- tf.reduce_sum(y * tf.log(hypothesis), axis=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001).minimize(loss)

predicted = tf.cast(tf.argmax(hypothesis,1), dtype=tf.float32)

accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y_true), dtype=tf.float32))


with tf.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for step in range(2001):
        sess.run(optimizer, feed_dict={x:x_train, y:y_train})

        if step % 100 == 0: 

            print(step, '\tloss :', sess.run(loss, feed_dict={x:x_train, y:y_train}))

    # predict
    for count, (i, j) in enumerate(zip(sess.run(y_true, feed_dict={y_true:y_test}), sess.run(predicted, feed_dict={x:x_test}))):
        print('y_true :', i, "\ty_pred :", j)
        if count == 20:
            break

    print('Acc :', accuracy_score(sess.run(y_true, feed_dict={y_true:y_test}), sess.run(predicted, feed_dict={x:x_test})))
    # print('Acc :', sess.run(accuracy, feed_dict={x:x_test, y_true:y_test}))