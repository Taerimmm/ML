# 이진분류
from sklearn.datasets import load_breast_cancer
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
tf.compat.v1.set_random_seed(42)

x = tf.placeholder(tf.float32, shape=[None,30])
y = tf.placeholder(tf.float32, shape=[None,1])

# 실습 
dataset = load_breast_cancer()
x_data = dataset.data
y_data = dataset.target.reshape(-1,1)
print(x_data.shape, y_data.shape)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

w = tf.Variable(tf.random_normal([30,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(x,w) + b)

cost = - tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1 - hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)

accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for step in range(5001):
        sess.run(train, feed_dict={x:x_train, y:y_train})

        if step & 50 == 0:
            print(step, '\t loss', sess.run(cost, feed_dict={x:x_train, y:y_train}))

    print('Acc :', sess.run(accuracy, feed_dict={x:x_test, y:y_test}))
    print('Acc :', accuracy_score(y_test, sess.run(predicted, feed_dict={x:x_test})))