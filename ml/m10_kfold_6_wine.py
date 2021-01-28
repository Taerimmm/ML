import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression # 이진 분류
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# 1. 데이터
dataset = load_wine()
x = dataset.data
y = dataset.target

print(x.shape, y.shape)  # (178, 13), (178,)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=45)
kfold = KFold(n_splits=5, shuffle=True)

# 2. 모델
for i in [LinearSVC, SVC, KNeighborsClassifier, DecisionTreeClassifier, RandomForestClassifier]:
    print()
    model = i()

    # 훈련
    scores = cross_val_score(model, x_train, y_train, cv=kfold)
    print(i.__name__ + '\'s score(acc) :', scores)

'''
LinearSVC's score(acc)              : [0.79310345 0.79310345 0.85714286 0.85714286 0.78571429]

SVC's score(acc)                    : [0.62068966 0.65517241 0.78571429 0.67857143 0.78571429]

KNeighborsClassifier's score(acc)   : [0.62068966 0.55172414 0.78571429 0.53571429 0.89285714]

DecisionTreeClassifier's score(acc) : [0.86206897 0.89655172 0.75       0.96428571 0.85714286]

RandomForestClassifier's score(acc) : [0.96551724 0.96551724 1.         1.         1.        ]
'''