import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

(x_train, _), (x_test, _) = mnist.load_data()

x = np.append(x_train, x_test, axis=0)

print(x.shape)

# 실습
# pca 를 통해 0.95 이상인 것 몇개?
# pca 배운거 다 집어넣고 확인

x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])

pca = PCA()
x = pca.fit_transform(x)

cumsum = np.cumsum(pca.explained_variance_ratio_)
print(cumsum)

d = np.argmax(cumsum >= 0.95) + 1
print('축소된 차원 수 :', d)

plt.plot(cumsum)
plt.grid()
plt.show()

pca = PCA(n_components=d)
x = pca.fit_transform(x)
print(x.shape)

x_train, x_test = train_test_split(x, test_size=1/7, random_state=45)

print(x_train.shape, x_test.shape)
