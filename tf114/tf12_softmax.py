import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(42)

x_data = [[1,2,1,1],
          [2,1,3,2],
          [3,1,3,4],
          [4,1,5,5],
          [1,7,5,5],
          [1,2,5,6],
          [1,6,6,6],
          [1,7,6,7]]
y_data = [[0,0,1],  # 2
          [0,0,1],
          [0,0,1],
          [0,1,0],  # 1
          [0,1,0],
          [0,1,0],
          [1,0,0],  # 0
          [1,0,0]]

x = tf.placeholder('float', shape=[None,4])
y = tf.placeholder('float', shape=[None,3])

w = tf.Variable(tf.random_normal([4,3]), name='weight')
b = tf.Variable(tf.random_normal([1,3]), name='bias')

hypothesis = tf.nn.softmax(tf.matmul(x,w) + b)

# categorical_cross_entropy
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for step in range(2001):
        _, cost_val = sess.run([optimizer, loss], feed_dict={x:x_data, y:y_data})
        if step % 200 == 0:
            print(step, cost_val)
    
    # predict
    a = sess.run(hypothesis, feed_dict={x:[[1,11,7,9]]})
    print(a, sess.run(tf.argmax(a,1)))