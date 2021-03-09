# 실습
import tensorflow as tf

node1 = tf.constant(2.0)
node2 = tf.constant(3.0)

add = tf.add(node1, node2)
sub = tf.subtract(node1, node2)
mul = tf.multiply(node1, node2)
div = tf.div(node1, node2)
remainder = tf.truncatediv

sess = tf.Session()

print('Add :', sess.run(add))
print('Subtract :', sess.run(sub))
print('Multiply :', sess.run(mul))
print('Div :', sess.run(div))