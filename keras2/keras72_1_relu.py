import numpy as np
import matplotlib.pyplot as plt

# ReLU(rectified linear unit)
def relu(x):
    return np.maximum(0, x)

x = np.arange(-5, 5, 0.1)
y = relu(x)

print(x)
print(y)

plt.plot(x, y)
plt.grid()
plt.show()
