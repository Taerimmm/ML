import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA

datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)     # (442, 10) (442,)

pca = PCA(n_components=7)

x2 = pca.fit_transform(x)
print(x2)
print(x2.shape)             # (442, 7)

pca_EVR = pca.explained_variance_ratio_
print(pca_EVR)
print(sum(pca_EVR))     

# 7개일 때 0.9479436357350414 -> 압축률이 94%
# 8개일 때 0.9913119559917797 -> 압축률이 99.1%
# 9개일 때 0.9991439470098977 -> 압축률이 99.9%

'''
통상적으로 95%이면 성능이 비슷하다 

판단은 알아서 
'''