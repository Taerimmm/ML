import tensorflow as tf

sess = tf.Session()

x = tf.Variable([2], dtype=tf.float32, name='test')

init = tf.global_variables_initializer() # 변수 초기화

sess.run(init)

print(sess.run(x))