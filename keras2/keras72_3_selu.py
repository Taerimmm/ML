import numpy as np
import matplotlib.pyplot as plt

# SeLU(scaled exponential linear unit)
def selu(x):
    alpha = 1.67326324
    scale = 1.05070098
    if not scale > 1:
        raise ValueError

    if x > 0:
        return scale * x
    else :
        return scale * alpha * (np.exp(x) - 1)

x = np.arange(-5, 5, 0.1)
y = [selu(i) for i in x]

print(x)
print(y)

plt.plot(x, y)
plt.grid()
plt.show()


#### 과제
# elu, selu, reakly_relu
# 72_2, 3, 4