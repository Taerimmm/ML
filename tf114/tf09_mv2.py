import tensorflow as tf
tf.compat.v1.set_random_seed(66)

x_data = [[73,51,65],
          [92,98,11],
          [89,31,33],
          [99,33,100],
          [17,66,79]]
y_data = [[152],
          [185],
          [180],
          [205],
          [142]]

x = tf.placeholder(tf.float32, shape=[None,3])
y = tf.placeholder(tf.float32, shape=[None,1])

w = tf.Variable(tf.random_normal([3,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# hypothesis = x * w + b
hypothesis = tf.matmul(x, w) + b

# 실습
# verbose 로 나오는 것은 step 과 cost 와 hypothesis

cost = tf.reduce_mean(tf.square(hypothesis - y))
train = tf.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for step in range(2001):
        sess.run(train, feed_dict={x:x_data, y:y_data})

        if step % 20 == 0:
            print(step, sess.run(cost, feed_dict={x:x_data, y:y_data}), '\n', sess.run(hypothesis, feed_dict={x:x_data, y:y_data}))
