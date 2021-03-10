# 실습
# placeholder 사용

import tensorflow as tf

tf.set_random_seed(66)

# x_train = [1,2,3]
# y_train = [3,5,7]

x_train = tf.placeholder(tf.float32, shape=[None])
y_train = tf.placeholder(tf.float32, shape=[None])

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = x_train * W + b

cost = tf.reduce_mean(tf.square(hypothesis - y_train))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

dic = {x_train:[1,2,3], y_train:[3,5,7]}
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2001):
        # sess.run(train)
        sess.run(train, feed_dict={x_train:[1,2,3], y_train:[3,5,7]})
        if step % 20 == 0:
            print(step, sess.run(cost, feed_dict=dic), sess.run(W), sess.run(b))

    print('Predict [4] :', sess.run(hypothesis, feed_dict={x_train:[4]}))
    print('Predict [5,6] :', sess.run(hypothesis, feed_dict={x_train:[5,6]}))
    print('Predict [6,7,8] :', sess.run(hypothesis, feed_dict={x_train:[6,7,8]}))

# 1. [4]
# 2. [5, 6]
# 3. [6, 7, 8]

