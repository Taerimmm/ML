# pip install lightgbm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from lightgbm import LGBMClassifier

from sklearn.metrics import precision_score

# Importing the dataset
X = np.load('./project/mini/data/X.npy')
y = pd.read_csv('./project/mini/data/y_label.csv', header=0).iloc[:,0]

X = X.reshape(X.shape[0], X.shape[1]*X.shape[2])

# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42, stratify=y)

# Feature Scaling
x_train /= -80
x_test /= -80


model = LGBMClassifier(objective='multiclass')

model.fit(x_train, y_train, categorical_feature=[0,12])

print('feature_importances :', model.feature_importances_)

y_pred = model.predict(x_test)
print('최종 정답률 :', model.score(x_test, y_test))

# 최종 정답률 : 0.5326016785022595
