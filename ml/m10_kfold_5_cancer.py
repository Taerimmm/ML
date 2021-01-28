import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression # 이진 분류
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# 1. 데이터
dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

print(x.shape, y.shape)  # (569, 30), (569,)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=45)
kfold = KFold(n_splits=5, shuffle=True)

# 2. 모델
for i in [LinearSVC, SVC, KNeighborsClassifier, LogisticRegression, DecisionTreeClassifier, RandomForestClassifier]:
    print()
    model = i()

    # 훈련
    scores = cross_val_score(model, x_train, y_train, cv=kfold)
    print(i.__name__ + '\'s score(acc) :', scores)

'''
LinearSVC's score(acc)              : [0.94505495 0.85714286 0.89010989 0.9010989  0.9010989 ]

SVC's score(acc)                    : [0.94505495 0.87912088 0.89010989 0.94505495 0.92307692]

KNeighborsClassifier's score(acc)   : [0.94505495 0.86813187 0.95604396 0.92307692 0.92307692]

LogisticRegression's score(acc)     : [0.97802198 0.93406593 0.9010989  0.91208791 0.95604396]

DecisionTreeClassifier's score(acc) : [0.94505495 0.84615385 0.93406593 0.93406593 0.87912088]

RandomForestClassifier's score(acc) : [0.98901099 0.96703297 0.94505495 0.96703297 0.95604396]
'''