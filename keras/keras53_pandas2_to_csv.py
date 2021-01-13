import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

dataset = load_iris()

x = dataset.data
y = dataset.target

# df = pd.DataFrame(x, columns=dataset.feature_names)
df = pd.DataFrame(x, columns=dataset['feature_names'])

df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

# y 컬럼을 추가
df['Target'] = y

df.to_csv('../data/csv/iris_sklearn.csv', sep=',')