import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
dataset = load_iris()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=45)

kfold = KFold(n_splits=5, shuffle=True)

# 2. 모델
for i in [MinMaxScaler, StandardScaler]:
    model = Pipeline([('MinMax', i()), ('RandomForest', RandomForestClassifier())])

    model.fit(x_train, y_train)

    result = model.score(x_test, y_test)
    print(i.__name__ + '\'s score :' ,result)

'''
MinMaxScaler's score : 0.9666666666666667
StandardScaler's score : 0.9333333333333333
'''