import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA

datasets = load_wine()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)     # (178, 13) (178,)

pca = PCA()
pca.fit(x)

cumsum = np.cumsum(pca.explained_variance_ratio_)
print('cumsum :', cumsum)

d = np.argmax(cumsum >= 0.95) + 1
print('cumsum >= 0.95 :', cumsum >= 0.95)
print('d :', d)

'''
차원을 얼마나 줄일 것인지 판단하는데 도움을 주는 역할
n_components 를 d 로 해주면 원하는 기준의 성능을 이끌어 낼 수 있다.
'''

plt.plot(cumsum)
plt.grid()
plt.show()