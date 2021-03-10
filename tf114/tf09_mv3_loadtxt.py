import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(66)

dataset = np.loadtxt('../data/csv/data-01-test-score.csv', dtype=float, delimiter=',')
print(dataset)
print(dataset.shape)

x_data = dataset[:,:-1]
y_data = dataset[:,-1].reshape(-1,1)

print(x_data.shape, y_data.shape)

x = tf.placeholder(tf.float32, shape=[None,3])
y = tf.placeholder(tf.float32, shape=[None,1])

w = tf.Variable(tf.random_normal([3,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(x,w) + b

cost = tf.reduce_mean(tf.square(hypothesis - y))
train = tf.train.GradientDescentOptimizer(learning_rate=0.0000508).minimize(cost)

sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())
for step in range(2001):
    sess.run(train, feed_dict={x:x_data, y:y_data})
    if step % 20 == 0:
        print(step, sess.run(cost, feed_dict={x:x_data, y:y_data})) 


test_data = [73, 80, 75, 152,
             93, 88, 93, 185,
             89, 91, 90, 180,
             96, 98, 100, 196,
             73, 66, 70, 142]
test_data = np.array(test_data).reshape(5,4)
x_test = test_data[:,:-1]
y_test = test_data[:,-1]

for i, j in enumerate(sess.run(hypothesis, feed_dict={x:x_test})):
    print('y_test :', y_test[i], '\ty_pred :', j)
sess.close()

# predict
# 73, 80, 75, 152
# 93, 88, 93, 185
# 89, 91, 90, 180
# 96, 98, 100, 196
# 73, 66, 70, 142