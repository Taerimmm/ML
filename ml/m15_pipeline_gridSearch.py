import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.svm import LinearSVC, SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression # 분류
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier

import warnings

warnings.filterwarnings('ignore')

# 1. 데이터
dataset = load_iris()
x = dataset.data
y = dataset.target

print(x.shape, y.shape)  # (150, 4) ,(150,)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=44)

kfold = KFold(n_splits=5, shuffle=True)

# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

parameters = [{"svc__C":[1, 10, 100, 1000], "svc__kernel":['linear']},
              {"svc__C":[1, 10, 100], 'svc__kernel':['rbf'], 'svc__gamma':[0.001, 0.0001]},
              {"svc__C":[1, 10, 100, 1000], 'svc__kernel':['sigmoid'], 'svc__gamma':[0.001, 0.0001]}]

parameters = [{"mal__C":[1, 10, 100, 1000], "mal__kernel":['linear']},
              {"mal__C":[1, 10, 100], 'mal__kernel':['rbf'], 'mal__gamma':[0.001, 0.0001]},
              {"mal__C":[1, 10, 100, 1000], 'mal__kernel':['sigmoid'], 'mal__gamma':[0.001, 0.0001]}]


# 2. 모델
pipe = Pipeline([("scaler", MinMaxScaler()), ('mal', SVC())])
# pipe = make_pipeline(MinMaxScaler(), SVC())

model = GridSearchCV(pipe, parameters, cv=kfold)


model.fit(x_train, y_train)

result = model.score(x_test, y_test)
print(result)