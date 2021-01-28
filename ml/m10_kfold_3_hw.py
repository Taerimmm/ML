# 실습 or 과제 !!!!
# train, test 나눈 다음에 발리데이션 하지 말고
# kfold 한 후에 train_test_split 사용

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression # 분류
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
dataset = load_iris()
x = dataset.data
y = dataset.target

print(x.shape, y.shape)  # (150, 4) ,(150,)

kfold = KFold(n_splits=5, shuffle=True,)

print(kfold.split(x,y))
# 2. 모델
model = LinearSVC()

i = 0
for train_index, test_index in kfold.split(x,y):
    # print(x[train_index], x[test_index])
    # print(y[train_index], y[test_index])
    i += 1
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # x_train, x_test, y_train, y_test = train_test_split(x[train_index], y[test_index], test_size=0.2)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

    
    scores = cross_val_score(model, x_train, y_train, cv=kfold)
    print(i,'time scores :', scores)
