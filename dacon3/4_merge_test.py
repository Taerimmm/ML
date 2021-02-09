import numpy as np
import time

start = time.time()

x_train = np.load('./dacon3/data/x_train_merge_1.npy')
for i in range(1,10):
    a = np.load('./dacon3/data/x_train_merge_{}.npy'.format(i+1))
    print(a.shape)
    x_train = np.append(x_train, a, axis=0)
print(x_train.shape)

x_train = x_train.reshape(50000, 256, 256, 1)
print(x_train.shape)
# (50000, 256, 256, 1)

x_test = np.load('./dacon3/data/x_test_merge.npy')
print(x_test.shape)

x_test = x_test.reshape(5000, 256, 256, 1)
print(x_test.shape)
# (5000, 256, 256, 1)

print("time :", time.time() - start)
# time : 415.61481642723083
