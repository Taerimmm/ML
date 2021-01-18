import numpy as np
import pandas as pd

def split_x(data,size):
    a = []
    for i in range(data.shape[0] - size + 1):
        a.append(np.array(data.iloc[i:(i+size), 0:len(data.columns)]))
    return  np.array(a)
dataset = pd.DataFrame(split_x(pd.DataFrame(range(1,11)),6).reshape(5,6))
print(dataset)

def split_x(data, x_len, y_len):
    a = []
    b = []
    size = 3
    for i in range(data.shape[0] - size + 1):
        a.append(np.array(data.iloc[i:size+i,:x_len]))
        b.append(np.array(data.iloc[size-1+i,-y_len:]))
    return np.array(a), np.array(b)

x_len = 4
y_len = 2
(X, Y) = split_x(dataset,x_len,y_len)
print(X)
print(Y)
print(X.shape, Y.shape)