import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression # 분류
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# 1. 데이터
dataset = load_wine()
x = dataset.data
y = dataset.target

print(x.shape)  # (150, 4)
print(y.shape)  # (150,)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

for j in [MinMaxScaler, StandardScaler]:
    scaler = j()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    print(j.__name__)
    # 2. 모델
    for i in [LinearSVC, SVC, KNeighborsClassifier, LogisticRegression, DecisionTreeClassifier, RandomForestClassifier]:
        print()
        model = i()

        # 훈련
        model.fit(x_train,y_train)

        y_pred = model.predict(x_test)
        # print('y_test :', y_test)
        # print('y_pred :', y_pred)

        result = model.score(x_test,y_test)
        print(i.__name__ + '\'s score(acc) :', result)
        
        acc = accuracy_score(y_test, y_pred)
        print(i.__name__ + '\'s accuracy_score :', acc)

    if j == StandardScaler:
        break
    print('=================================================================')

'''
MinMaxScaler

LinearSVC's score(acc) : 0.9722222222222222
LinearSVC's accuracy_score : 0.9722222222222222

SVC's score(acc) : 1.0
SVC's accuracy_score : 1.0

KNeighborsClassifier's score(acc) : 0.9722222222222222
KNeighborsClassifier's accuracy_score : 0.9722222222222222

LogisticRegression's score(acc) : 1.0
LogisticRegression's accuracy_score : 1.0

DecisionTreeClassifier's score(acc) : 0.9166666666666666
DecisionTreeClassifier's accuracy_score : 0.9166666666666666

RandomForestClassifier's score(acc) : 1.0
RandomForestClassifier's accuracy_score : 1.0
=================================================================
StandardScaler

LinearSVC's score(acc) : 0.9722222222222222
LinearSVC's accuracy_score : 0.9722222222222222

SVC's score(acc) : 1.0
SVC's accuracy_score : 1.0

KNeighborsClassifier's score(acc) : 0.9722222222222222
KNeighborsClassifier's accuracy_score : 0.9722222222222222

LogisticRegression's score(acc) : 1.0
LogisticRegression's accuracy_score : 1.0

DecisionTreeClassifier's score(acc) : 0.9166666666666666
DecisionTreeClassifier's accuracy_score : 0.9166666666666666

RandomForestClassifier's score(acc) : 1.0
RandomForestClassifier's accuracy_score : 1.0
'''

'''
Tensorflow's acc : 0.9722222089767456
'''