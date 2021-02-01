import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA

datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)     # (442, 10) (442,)

# pca = PCA(n_components=7)

# x2 = pca.fit_transform(x)
# print(x2)
# print(x2.shape)             # (442, 7)

# pca_EVR = pca.explained_variance_ratio_
# print(pca_EVR)
# print(sum(pca_EVR))     

# 7개일 때 0.9479436357350414 -> 압축률이 94%
# 8개일 때 0.9913119559917797 -> 압축률이 99.1%
# 9개일 때 0.9991439470098977 -> 압축률이 99.9%

pca = PCA()
pca.fit(x)

cumsum = np.cumsum(pca.explained_variance_ratio_)
print('cumsum :', cumsum)

d = np.argmax(cumsum >= 0.95) + 1
print('cumsum >= 0.95 :', cumsum >= 0.95)
print('d :', d)

'''
차원을 얼마나 줄일 것인지 판단하는데 도움을 주는 역할
n_components 를 d 로 해주면 해당 값의 성능을 이끌어 낼 수 있다.
'''

plt.plot(cumsum)
plt.grid()
plt.show()