import tensorflow as tf

x = [1,2,3]
W = tf.Variable([0.3], tf.float32)
b = tf.Variable([1.0], tf.float32)

hypothesis = W * x + b

# ...
# print('hypothesis :', ???)

# 실습
# 1. sess.run()
# 2. InteractiveSession
# 3. .eval(session=sess)
# hypothesis를 출력하는 코드

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
aaa = sess.run(hypothesis)
print('hypothesis_1 :', aaa)
sess.close()

sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())
bbb = hypothesis.eval()
print('hypothesis_2 :', bbb)
sess.close()

sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())
ccc = hypothesis.eval(session=sess)
print('hypothesis_3 :', ccc)
sess.close()