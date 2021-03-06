import numpy as np
import matplotlib.pyplot as plt

# ELU(exponential linear unit)
def elu(x):
    alpha = 1.0
    if not alpha > 0:
        raise ValueError

    if x > 0:
        return x
    else :
        return alpha * (np.exp(x) - 1)

x = np.arange(-5, 5, 0.1)
y = [elu(i) for i in x]

print(x)
print(y)

plt.plot(x, y)
plt.grid()
plt.show()


#### 과제
# elu, selu, reakly_relu
# 72_2, 3, 4