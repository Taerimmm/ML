from sklearn.datasets import load_boston
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
tf.compat.v1.set_random_seed(42)

x = tf.placeholder(tf.float32, shape=[None,13])
y = tf.placeholder(tf.float32, shape=[None,1])

# 실습 
dataset = load_boston()
x_data = dataset.data
y_data = dataset.target.reshape(-1,1)
print(x_data.shape)
print(y_data.shape)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

w = tf.Variable(tf.random_normal([13,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(x,w) + b

cost = tf.reduce_mean(tf.square(hypothesis - y))

train = tf.train.GradientDescentOptimizer(learning_rate=0.267).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    
    for step in range(5001):
        sess.run(train, feed_dict={x:x_train, y:y_train})

        if step % 50 == 0:
            print(step, '\tloss :',sess.run(cost, feed_dict={x:x_train, y:y_train}))

    print('R2 :', r2_score(y_test, sess.run(hypothesis, feed_dict={x:x_test})))
