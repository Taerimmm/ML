import numpy as np
import pandas as pd

def split_x(data,size):
    a = []
    for i in range(data.shape[0] - size + 1):
        a.append(np.array(data.iloc[i:(i+size), 0:len(data.columns)]))
    return  np.array(a)
dataset = pd.DataFrame(split_x(pd.DataFrame(range(1,21)),6).reshape(15,6))
print(dataset)

def split_x(data, x_row, x_col, y_row, y_col):
    a, b =[], []
    x_step, y_step = 0, 0
    for i in range(data.shape[0] - x_row + 1):
        x_step += 1
        a.append(np.array(data.iloc[i:i+x_row,:x_col]))

    for i in range(data.shape[0] - y_row + 1):
        if x_row + y_row+ i > data.shape[0]:
            break
        y_step += 1
        b.append(np.array(data.iloc[x_row+i:x_row+i+y_row,-y_col:]))

    a = np.array(a)[:y_step]
    b = np.array(b)

    return  a, b

x_row, x_col = 5, 3
y_row, y_col = 4, 2
(X, Y) = split_x(dataset, x_row, x_col, y_row, y_col)
print(X)
print(Y)
print(X.shape, Y.shape)