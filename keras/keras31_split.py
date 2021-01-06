import numpy as np

a = np.array(range(1,11))
size = 3

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)]
        # aaa.append([item for item in subset])
        aaa.append(subset)
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(a, size)
print("==========================")
print(dataset)

# a = np.array(range(1,11))
# size = 4

# def split_x(seq, size):
#     return np.array([a[i:(i+size)] for i in [i for i in range(len(a) - size + 1)]])

# dataset = split_x(a, size)
# print("==========================")
# print(dataset)