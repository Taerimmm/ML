import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

tf.compat.v1.set_random_seed(42)

dataset = load_iris()

x_data = dataset.data
y_data = dataset.target.reshape(-1,1)
print(x_data.shape, y_data.shape) # (150, 4), (150, 1)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

y_test_ = y_test.copy()

ohe = OneHotEncoder()
y_train = ohe.fit_transform(y_train).toarray()
y_test = ohe.transform(y_test).toarray()
print(y_train)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x = tf.placeholder(tf.float32, shape=[None,4])
y = tf.placeholder(tf.float32, shape=[None,3])

y_true = tf.placeholder(tf.float32, shape=[None,1])

w = tf.Variable(tf.random_normal([4,3]), name='weight')
b = tf.Variable(tf.random_normal([1,3]), name='bias')

hypothesis = tf.nn.softmax(tf.matmul(x,w) + b)

loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis=1))
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-2).minimize(loss)

predicted = tf.argmax(hypothesis,1)

accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, tf.argmax(y,1)), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for step in range(5001):
        sess.run(optimizer, feed_dict={x:x_train, y:y_train})
        if step % 200 == 0:
            print(step, '\tloss :', sess.run(loss, feed_dict={x:x_train, y:y_train}))

    for i, j in zip(sess.run(y_true, feed_dict={y_true:y_test_}), sess.run(predicted, feed_dict={x:x_test})):
        print('y_test :', i[0], '\ty_pred :', j)

    print('Acc :', sess.run(accuracy, feed_dict={x:x_test, y:y_test}))
    print('Acc :', accuracy_score(sess.run(y_true, feed_dict={y_true:y_test_}), sess.run(predicted, feed_dict={x:x_test})))
